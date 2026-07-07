from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.services.news_service import news_service
from app.services.rag_service import create_rag_service, RAGService
from app.services.cache_service import get_cache_service
from app.core.config import get_settings, Settings

router = APIRouter(prefix="/report", tags=["AI Report"])


# ─────────────────────────────────────────
# Request/Response 스키마
# ─────────────────────────────────────────

class ReportContext(BaseModel):
    sales_delta: float | None = None
    foot_traffic_delta: float | None = None
    store_count_delta: float | None = None
    closure_rate: float | None = None
    vacancy_rate: float | None = None
    top_age_group: str | None = None
    top_gender: str | None = None


class ReportGenerateRequest(BaseModel):
    region_code: str
    region_name: str
    district_name: str
    year: int
    quarter: int
    decline_grade: str
    score: float
    context: ReportContext


class CauseItem(BaseModel):
    title: str
    level: str
    description: str


class SignalItem(BaseModel):
    title: str
    description: str


class SimilarCaseItem(BaseModel):
    region_code: str
    summary: str
    description: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    tags: list[str] = []


class AlternativeRegionItem(BaseModel):
    region_code: str
    reason: str
    stat: str


class DecisionReasons(BaseModel):
    reason_1: str | None = None
    reason_2: str | None = None
    reason_3: str | None = None


class ReportGenerateResponse(BaseModel):
    ai_outlook: str
    summary: str
    decision_recommendation: str
    decision_title: str
    decision_description: str
    causes: list[CauseItem]
    signals: list[SignalItem]
    decision_reasons: DecisionReasons
    similar_cases: list[SimilarCaseItem]
    alternative_regions: list[AlternativeRegionItem]


class NewsSearchRequest(BaseModel):
    query: str
    max_results: int = 10


class EmbedRequest(BaseModel):
    texts: list[str]
    metadatas: list[dict] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


# ─────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────

def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return create_rag_service(openai_api_key=settings.openai_api_key)


# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────

@router.post("/generate", response_model=ReportGenerateResponse)
async def generate_report(
    request: ReportGenerateRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    AI 리포트 생성
    - 뉴스 검색 + RAG 임베딩
    - LLM으로 분석 생성
    - all_grades 캐시 필수 (없으면 400)
    """
    cache = get_cache_service()

    # all_grades 캐시 확인 (유사사례/대안지역용)
    all_grades = cache.get_all_grades(request.year, request.quarter)
    if all_grades is None:
        raise HTTPException(
            status_code=400,
            detail=f"등급 데이터 없음. POST /api/grade/batch를 먼저 실행하세요. (year={request.year}, quarter={request.quarter})"
        )

    # 뉴스 검색 쿼리 생성
    news_query = f"{request.district_name} 상권"

    # 캐시 확인 후 뉴스 임베딩
    if not cache.is_news_embedded(news_query):
        # 뉴스 검색
        articles = await news_service.search_news(
            query=news_query,
            max_results=10
        )

        if articles:
            # 임베딩
            texts = [f"{a['title']} {a['description']}" for a in articles]
            metadatas = [{"source": "news", "query": news_query} for _ in articles]
            rag_service.add_documents(texts=texts, metadatas=metadatas)

            # 캐시 플래그
            cache.set_news_embedded(news_query)

    # RAG 컨텍스트 생성
    rag_context = rag_service.generate_context(
        query=f"{request.region_name} {request.district_name} 상권 {request.decline_grade}",
        k=3
    )

    # TODO: LLM 호출해서 실제 리포트 생성
    # 지금은 목업 응답

    return ReportGenerateResponse(
        ai_outlook=f"{request.region_name}은(는) 현재 {request.decline_grade} 등급으로 분류됩니다. (RAG 컨텍스트: {len(rag_context)}자)",
        summary=f"{request.district_name} {request.region_name} 상권 분석 요약",
        decision_recommendation="HOLD",
        decision_title="현상 유지 권고",
        decision_description="추가 분석이 필요합니다.",
        causes=[
            CauseItem(title="유동인구 변화", level="MEDIUM", description="분석 중...")
        ],
        signals=[
            SignalItem(title="임대료 동향", description="분석 중...")
        ],
        decision_reasons=DecisionReasons(
            reason_1="시장 상황 관망",
            reason_2="추가 데이터 필요",
            reason_3=None
        ),
        similar_cases=[],  # TODO: all_grades 기반 유사 지역 찾기
        alternative_regions=[]  # TODO: all_grades 기반 대안 지역 찾기
    )


@router.post("/news/search")
async def search_news(request: NewsSearchRequest):
    """뉴스 검색"""
    articles = await news_service.search_news(
        query=request.query,
        max_results=request.max_results
    )
    return {"articles": articles, "count": len(articles)}


@router.post("/embed")
def embed_documents(
    request: EmbedRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """문서 벡터 임베딩"""
    try:
        count = rag_service.add_documents(
            texts=request.texts,
            metadatas=request.metadatas
        )
        return {"message": f"{count} chunks embedded", "chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_similar(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """유사 문서 검색"""
    try:
        results = rag_service.search(query=request.query, k=request.k)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context")
def generate_context(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """RAG 컨텍스트 생성"""
    try:
        context = rag_service.generate_context(query=request.query, k=request.k)
        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
