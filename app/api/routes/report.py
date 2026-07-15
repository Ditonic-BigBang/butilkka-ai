from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional
from app.services.report_service import create_report_service, ReportService
from app.core.config import get_settings, Settings

router = APIRouter(prefix="/report", tags=["AI Report"])


# ─────────────────────────────────────────
# Request/Response 스키마 (ERD 기준)
# ─────────────────────────────────────────

class AliasedModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ReportContext(AliasedModel):
    """Spring에서 전달하는 상권 지표"""
    sales_delta: float | None = Field(default=None, alias="salesDelta")          # 매출 변화율
    foot_traffic_delta: float | None = Field(default=None, alias="footTrafficDelta")   # 유동인구 변화율
    store_count_delta: float | None = Field(default=None, alias="storeCountDelta")    # 점포수 변화율
    closure_rate: float | None = Field(default=None, alias="closureRate")         # 폐업률
    vacancy_rate: float | None = Field(default=None, alias="vacancyRate")         # 공실률
    top_age_group: str | None = Field(default=None, alias="topAgeGroup")          # 주요 연령대
    top_gender: str | None = Field(default=None, alias="topGender")             # 주요 성별


class QuarterlyMetrics(AliasedModel):
    """8분기 이력 (오래된→최근 순). Spring이 commercial_stats에서 조회해서 전달.
    없으면 다음 분기 예측은 스킵되고 나머지 리포트는 그대로 생성됨."""
    sales_qoq: list[float] = Field(alias="salesQoq")
    foot_traffic: list[float] = Field(alias="footTraffic")
    store_count: list[float] = Field(alias="storeCount")
    closure_rate: list[float] = Field(alias="closureRate")


class ReportGenerateRequest(AliasedModel):
    """리포트 생성 요청"""
    region_code: str = Field(alias="regionCode")      # 구 코드 (5자리)
    region_name: str = Field(alias="regionName")      # 구 이름
    district_name: str | None = Field(default=None, alias="districtName")  # 구 이름
    year: int
    quarter: int
    grade: str               # A~E
    score: int               # 0~100
    decline_type: str = Field(alias="declineType")   # 성장/정체/쇠퇴
    context: ReportContext
    quarterly_history: QuarterlyMetrics | None = Field(default=None, alias="quarterlyHistory")

    @model_validator(mode="before")
    @classmethod
    def normalize_region_fields(cls, values):
        if not isinstance(values, dict):
            return values

        region_code = values.get("regionCode") or values.get("region_code")
        if isinstance(region_code, str):
            region_code = region_code.strip()
            if len(region_code) == 10 and region_code.isdigit():
                values["regionCode"] = region_code[:5]
                values["region_code"] = region_code[:5]

        region_name = values.get("regionName") or values.get("region_name")
        district_name = values.get("districtName") or values.get("district_name")
        if not district_name and region_name:
            values["districtName"] = region_name
            values["district_name"] = region_name

        return values


# ─── 하위 테이블 스키마 ───

class CauseItem(AliasedModel):
    """report_cause 테이블"""
    title: str
    level: str          # 높음/중간/낮음
    description: str = ""  # 선행 원인은 title+level만 사용, 상세 설명 불필요


class SignalItem(AliasedModel):
    """report_signal 테이블"""
    title: str
    description: str = ""  # 선행 신호는 title 한 줄만 사용, 상세 설명 불필요


class DecisionReasons(AliasedModel):
    """report_decision_reasons 테이블"""
    reason_1: str | None = None
    reason_2: str | None = None
    reason_3: str | None = None


class SimilarCaseItem(AliasedModel):
    """report_similar_cases 테이블"""
    region_code: str = Field(alias="regionCode")
    region_name: str = Field(default="", alias="regionName")
    summary: str
    description: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    tag1: str | None = None
    tag2: str | None = None
    tag3: str | None = None
    tag4: str | None = None


class AlternativeRegionItem(AliasedModel):
    """report_alternative_regions 테이블"""
    region_code: str = Field(alias="regionCode")
    region_name: str = Field(default="", alias="regionName")
    reason: str
    stat: str


class ReportGenerateResponse(AliasedModel):
    """리포트 생성 응답 (reports 테이블 + 하위 테이블)"""
    # reports 테이블 필드
    summary: str                      # 한 줄 요약
    ai_outlook: str = Field(alias="aiOutlook")                   # AI 종합 전망 (5~6줄)
    decision_recommendation: str = Field(alias="decisionRecommendation")      # 버티기/이동
    decision_title: str = Field(alias="decisionTitle")
    decision_description: str = Field(alias="decisionDescription")

    # 하위 테이블
    causes: list[CauseItem]
    signals: list[SignalItem]
    decision_reasons: DecisionReasons = Field(alias="decisionReasons")
    similar_cases: list[SimilarCaseItem] = Field(alias="similarCases")
    alternative_regions: list[AlternativeRegionItem] = Field(alias="alternativeRegions")

    # 다음 분기 예측 (quarterly_history 없으면 둘 다 null)
    predicted_trend: str | None = Field(default=None, alias="predictedTrend")       # 성장/유지/쇠퇴
    predicted_next_grade: str | None = Field(default=None, alias="predictedNextGrade")  # A~E


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
    - all_grades 캐시 있으면 대안지역 추천에 사용 (없어도 나머지는 정상 생성됨)
    """
    try:
        result = await report_service.generate(
            region_code=request.region_code,
            region_name=request.region_name,
            district_name=request.district_name or request.region_name,
            year=request.year,
            quarter=request.quarter,
            grade=request.grade,
            score=request.score,
            decline_type=request.decline_type,
            context=request.context.model_dump(),
            quarterly_history=request.quarterly_history.model_dump() if request.quarterly_history else None,
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
            alternative_regions=[AlternativeRegionItem(**a) for a in result.get("alternative_regions", [])],
            predicted_trend=result.get("predicted_trend"),
            predicted_next_grade=result.get("predicted_next_grade"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
