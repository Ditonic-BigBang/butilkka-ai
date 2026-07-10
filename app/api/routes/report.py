from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.services.report_service import create_report_service, ReportService
from app.services.cache_service import get_cache_service
from app.core.config import get_settings, Settings

router = APIRouter(prefix="/report", tags=["AI Report"])


# ─────────────────────────────────────────
# Request/Response 스키마 (ERD 기준)
# ─────────────────────────────────────────

class ReportContext(BaseModel):
    """Spring에서 전달하는 상권 지표"""
    sales_delta: float | None = None          # 매출 변화율
    foot_traffic_delta: float | None = None   # 유동인구 변화율
    store_count_delta: float | None = None    # 점포수 변화율
    closure_rate: float | None = None         # 폐업률
    vacancy_rate: float | None = None         # 공실률
    top_age_group: str | None = None          # 주요 연령대
    top_gender: str | None = None             # 주요 성별


class ReportGenerateRequest(BaseModel):
    """리포트 생성 요청"""
    region_code: str         # 상권 코드 (10자리)
    region_name: str         # 행정동 이름
    district_name: str       # 자치구 이름
    year: int
    quarter: int
    grade: str               # A~E
    score: int               # 0~100
    decline_type: str        # 성장/정체/쇠퇴
    context: ReportContext


# ─── 하위 테이블 스키마 ───

class CauseItem(BaseModel):
    """report_cause 테이블"""
    title: str
    level: str          # 높음/중간/낮음
    description: str


class SignalItem(BaseModel):
    """report_signal 테이블"""
    title: str
    description: str


class DecisionReasons(BaseModel):
    """report_decision_reasons 테이블"""
    reason_1: str | None = None
    reason_2: str | None = None
    reason_3: str | None = None


class SimilarCaseItem(BaseModel):
    """report_similar_cases 테이블"""
    region_code: str
    summary: str
    description: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    tag1: str | None = None
    tag2: str | None = None
    tag3: str | None = None
    tag4: str | None = None


class AlternativeRegionItem(BaseModel):
    """report_alternative_regions 테이블"""
    region_code: str
    reason: str
    stat: str


class ReportGenerateResponse(BaseModel):
    """리포트 생성 응답 (reports 테이블 + 하위 테이블)"""
    # reports 테이블 필드
    summary: str                      # 한 줄 요약
    ai_outlook: str                   # AI 종합 전망 (5~6줄)
    decision_recommendation: str      # 버티기/이동
    decision_title: str
    decision_description: str

    # 하위 테이블
    causes: list[CauseItem]
    signals: list[SignalItem]
    decision_reasons: DecisionReasons
    similar_cases: list[SimilarCaseItem]
    alternative_regions: list[AlternativeRegionItem]


# ─────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────

def get_report_service(settings: Settings = Depends(get_settings)) -> ReportService:
    return create_report_service(
        openai_api_key=settings.openai_api_key,
        chroma_db_dir=settings.chroma_db_dir,
    )


# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────

@router.post("/generate", response_model=ReportGenerateResponse)
async def generate_report(
    request: ReportGenerateRequest,
    report_service: ReportService = Depends(get_report_service)
):
    """
    AI 리포트 생성
    - Tavily 뉴스 검색
    - FAISS 임베딩
    - 3회 LLM 호출 (전망/요약, 원인/시그널, 의사결정)
    - all_grades 캐시로 유사사례/대안지역
    """
    cache = get_cache_service()

    # all_grades 캐시 확인
    all_grades = cache.get_all_grades(request.year, request.quarter)
    if all_grades is None:
        raise HTTPException(
            status_code=400,
            detail=f"등급 데이터 없음. POST /api/grade/batch를 먼저 실행하세요. (year={request.year}, quarter={request.quarter})"
        )

    try:
        result = await report_service.generate(
            region_code=request.region_code,
            region_name=request.region_name,
            district_name=request.district_name,
            year=request.year,
            quarter=request.quarter,
            grade=request.grade,
            score=request.score,
            decline_type=request.decline_type,
            context=request.context.model_dump()
        )

        return ReportGenerateResponse(
            summary=result.get("summary", ""),
            ai_outlook=result.get("ai_outlook", ""),
            decision_recommendation=result.get("decision_recommendation", "버티기"),
            decision_title=result.get("decision_title", ""),
            decision_description=result.get("decision_description", ""),
            causes=[CauseItem(**c) for c in result.get("causes", [])],
            signals=[SignalItem(**s) for s in result.get("signals", [])],
            decision_reasons=DecisionReasons(**result.get("decision_reasons", {})),
            similar_cases=[SimilarCaseItem(**s) for s in result.get("similar_cases", [])],
            alternative_regions=[AlternativeRegionItem(**a) for a in result.get("alternative_regions", [])]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
