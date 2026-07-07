import httpx
from bs4 import BeautifulSoup
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NewsService:
    """뉴스 API 크롤링 서비스"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search_news(
        self,
        query: str,
        max_results: int = 10
    ) -> list[dict]:
        """뉴스 검색 (네이버 뉴스 API 예시)"""
        # TODO: 실제 뉴스 API 연동 필요
        # 네이버, 카카오, NewsAPI 등 선택

        headers = {
            "X-Naver-Client-Id": self.api_key or "",
            "X-Naver-Client-Secret": "",  # config에서 가져오기
        }

        try:
            response = await self.client.get(
                "https://openapi.naver.com/v1/search/news.json",
                params={"query": query, "display": max_results},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("items", []):
                articles.append({
                    "title": self._clean_html(item.get("title", "")),
                    "description": self._clean_html(item.get("description", "")),
                    "link": item.get("link", ""),
                    "pub_date": item.get("pubDate", ""),
                })
            return articles

        except Exception as e:
            logger.error(f"뉴스 검색 실패: {e}")
            return []

    async def fetch_article_content(self, url: str) -> str:
        """뉴스 본문 크롤링"""
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 본문 추출 (사이트별로 다름)
            article = soup.find("article") or soup.find("div", class_="content")
            if article:
                return article.get_text(strip=True)
            return ""

        except Exception as e:
            logger.error(f"본문 크롤링 실패: {e}")
            return ""

    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(strip=True)

    async def close(self):
        await self.client.aclose()


# Singleton
news_service = NewsService()
