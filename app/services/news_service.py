from tavily import TavilyClient
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NewsService:
    """Tavily 뉴스 검색 서비스"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.client = TavilyClient(api_key=api_key)

    def _ensure_client(self):
        """클라이언트 초기화 확인"""
        if not self.client:
            from app.core.config import get_settings
            self.api_key = get_settings().tavily_api_key
            if self.api_key:
                self.client = TavilyClient(api_key=self.api_key)
            else:
                raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다")

    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        days: int = 90
    ) -> list[dict]:
        """Tavily로 뉴스/웹 검색 (최근 N일)"""
        try:
            self._ensure_client()

            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                days=days,
                include_answer=False,
                include_raw_content=False,
            )

            articles = []
            for item in response.get("results", []):
                articles.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                })

            logger.info(f"Tavily 검색 완료: '{query}' → {len(articles)}건")
            return articles

        except Exception as e:
            logger.error(f"Tavily 검색 실패: {e}")
            return []

    async def close(self):
        """리소스 정리 (Tavily는 별도 정리 불필요)"""
        pass


# Singleton (lazy init)
news_service = NewsService()
