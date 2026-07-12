from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_service import model_service
from app.services.cache_service import get_cache_service

router = APIRouter(prefix="/grade", tags=["Grade Prediction"])


# ─────────────────────────────────────────
# Request/Response 스키마 (ERD 기준)
# ─────────────────────────────────────────

class GradeItem(BaseModel):
    region_code: str
    year: int
    quarter: int
    sales_delta: float
    foot_traffic: int
    store_count: int | None = None
    closure_rate: float | None = None


class GradeBatchRequest(BaseModel):
    items: list[GradeItem]


class GradeResult(BaseModel):
    region_code: str
    year: int
    quarter: int
    grade: str           # A~E
    score: int           # 0~100
    decline_type: str    # 성장/정체/쇠퇴


class GradeBatchResponse(BaseModel):
    results: list[GradeResult]
    cached: bool = False


# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────

@router.get("/models")
def list_models():
    """사용 가능한 모델 목록"""
    models = model_service.list_models()
    return {"models": models}


@router.post("/batch", response_model=GradeBatchResponse)
def predict_batch(request: GradeBatchRequest):
    """
    등급 배치 추론
    - 휴리스틱으로 전체 지역 등급/점수/유형 계산 (모델 자동 갱신은 추후 작업)
    - Redis에 개별 등급 + all_grades 캐싱
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="items가 비어있습니다")

    try:
        # 결과 매핑 + 캐싱
        # NOTE: 등급/점수/유형은 당분간 아래 휴리스틱으로 계산 (모델 자동 갱신은 추후 작업)
        cache = get_cache_service()
        results = []
        all_grades = []

        for item in request.items:
            # score 계산 (0~100)
            score = _calculate_score(item.sales_delta, item.foot_traffic, item.closure_rate or 0)

            # grade 계산 (A~E)
            grade = _score_to_grade(score)

            # decline_type 계산 (성장/정체/쇠퇴)
            decline_type = _calculate_decline_type(item.sales_delta, item.foot_traffic)

            result = GradeResult(
                region_code=item.region_code,
                year=item.year,
                quarter=item.quarter,
                grade=grade,
                score=score,
                decline_type=decline_type
            )
            results.append(result)

            # 개별 캐시
            cache.set_grade(
                region_code=item.region_code,
                year=item.year,
                quarter=item.quarter,
                grade=grade,
                score=score,
                decline_type=decline_type
            )

            all_grades.append({
                "region_code": item.region_code,
                "grade": grade,
                "score": score,
                "decline_type": decline_type
            })

        # 전체 등급 캐시 (year/quarter 기준)
        if results:
            first = request.items[0]
            cache.set_all_grades(first.year, first.quarter, all_grades)

        return GradeBatchResponse(results=results, cached=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{region_code}/{year}/{quarter}")
def get_grade(region_code: str, year: int, quarter: int):
    """캐시된 등급 조회"""
    cache = get_cache_service()
    result = cache.get_grade(region_code, year, quarter)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="등급 없음. /grade/batch 먼저 실행하세요."
        )

    return {
        "region_code": region_code,
        "year": year,
        "quarter": quarter,
        **result
    }


# ─────────────────────────────────────────
# 헬퍼 함수 (임시 로직, 모델 완성 후 교체)
# ─────────────────────────────────────────

def _calculate_score(sales_delta: float, foot_traffic: int, closure_rate: float) -> int:
    """점수 계산 (0~100) - 임시 로직"""
    # 매출 변화율 기여 (40점)
    sales_score = min(max((sales_delta + 50) * 0.4, 0), 40)

    # 유동인구 기여 (40점)
    traffic_score = min(foot_traffic / 10000 * 40, 40)

    # 폐업률 감점 (최대 -20점)
    closure_penalty = min(closure_rate * 2, 20)

    score = int(sales_score + traffic_score - closure_penalty + 20)
    return max(0, min(100, score))


def _score_to_grade(score: int) -> str:
    """점수 → 등급 변환"""
    if score >= 80:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    elif score >= 20:
        return "D"
    else:
        return "E"


def _calculate_decline_type(sales_delta: float, foot_traffic: int) -> str:
    """쇠퇴 유형 계산 (성장/정체/쇠퇴)"""
    if sales_delta > 5:
        return "성장"
    elif sales_delta < -5:
        return "쇠퇴"
    else:
        return "정체"
