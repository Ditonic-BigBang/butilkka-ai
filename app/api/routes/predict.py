from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_service import model_service
from app.services.cache_service import get_cache_service

router = APIRouter(prefix="/grade", tags=["Grade Prediction"])


# ─────────────────────────────────────────
# Request/Response 스키마
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
    decline_grade: str
    score: float


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
    - pkl 모델로 전체 지역 등급 계산
    - Redis에 개별 등급 + all_grades 캐싱
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="items가 비어있습니다")

    try:
        # 피처 추출 (모델 입력용)
        features = []
        for item in request.items:
            features.append({
                "sales_delta": item.sales_delta,
                "foot_traffic": item.foot_traffic,
                "store_count": item.store_count or 0,
                "closure_rate": item.closure_rate or 0,
            })

        # 모델 예측
        predictions = model_service.predict(
            model_name="decline_grade",
            data=features
        )

        # 결과 매핑 + 캐싱
        cache = get_cache_service()
        results = []
        all_grades = []

        for item, pred in zip(request.items, predictions):
            # TODO: score 계산 로직 (모델에 따라 다름)
            grade = str(pred)
            score = 0.5  # 임시값, 모델 predict_proba 사용 시 수정

            result = GradeResult(
                region_code=item.region_code,
                year=item.year,
                quarter=item.quarter,
                decline_grade=grade,
                score=score
            )
            results.append(result)

            # 개별 캐시
            cache.set_grade(
                region_code=item.region_code,
                year=item.year,
                quarter=item.quarter,
                grade=grade,
                score=score
            )

            all_grades.append({
                "region_code": item.region_code,
                "grade": grade,
                "score": score
            })

        # 전체 등급 캐시 (year/quarter 기준)
        if results:
            first = request.items[0]
            cache.set_all_grades(first.year, first.quarter, all_grades)

        return GradeBatchResponse(results=results, cached=True)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"모델 없음: {e}")
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
