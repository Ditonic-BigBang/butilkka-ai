from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.news_service import news_service
from app.services.rag_service import create_rag_service, RAGService
from app.core.config import get_settings, Settings

router = APIRouter(prefix="/report", tags=["AI Report"])


class NewsSearchRequest(BaseModel):
    query: str
    max_results: int = 10


class EmbedRequest(BaseModel):
    texts: list[str]
    metadatas: list[dict] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return create_rag_service(openai_api_key=settings.openai_api_key)


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
