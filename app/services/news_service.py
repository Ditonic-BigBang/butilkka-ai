from tavily import TavilyClient
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class NewsService:
    """Tavily 뉴스 검색 서비스"""

    # 한국 주요 뉴스 도메인
    KR_NEWS_DOMAINS = [
        "news.naver.com",
        "n.news.naver.com",
        "yonhapnews.co.kr",
        "yna.co.kr",
        "chosun.com",
        "donga.com",
        "joongang.co.kr",
        "hani.co.kr",
        "khan.co.kr",
        "mk.co.kr",
        "edaily.co.kr",
        "sedaily.com",
        "newsis.com",
        "news1.kr",
        "mt.co.kr",
        "hankyung.com",
    ]

    # 상권 쇠퇴 선행 신호 카테고리 (구 단위 검색 키워드 접미사)
    CATEGORY_QUERIES = {
        "재개발/도시계획": "재개발",
        "교통 접근성": "지하철 노선",
        "대형 앵커시설": "대형시설 입점 폐점",
        "부동산/임대료": "임대료",
        "인구구조": "인구 감소",
        "정책/규제": "상권활성화구역",
        "경쟁 상권 부상": "신흥 상권",
        "안전/사고": "화재 치안",
    }

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
        days: int = 90,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Tavily로 뉴스/웹 검색 (최근 N일)"""
        try:
            self._ensure_client()

            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth="basic",
                max_results=max_results,
                days=days,
                include_domains=self.KR_NEWS_DOMAINS,
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
                    "category": category,
                })

            logger.info(f"Tavily 검색 완료: '{query}' → {len(articles)}건")
            return articles

        except Exception as e:
            logger.error(f"Tavily 검색 실패: {e}")
            return []

    async def search_by_categories(
        self,
        district_name: str,
        days: int = 90,
        max_total: int = 20,
    ) -> list[dict]:
        """구 단위로 일반 상권 검색 + 8개 선행 신호 카테고리 검색을 동시 실행 후 병합"""
        queries = [(f"{district_name} 상권", "일반", 5)]
        queries += [
            (f"{district_name} {keyword}", category, 3)
            for category, keyword in self.CATEGORY_QUERIES.items()
        ]

        results = await asyncio.gather(*[
            self.search_news(query=q, max_results=max_results, days=days, category=category)
            for q, category, max_results in queries
        ])

        seen_urls = set()
        merged = []
        for articles in results:
            for article in articles:
                url = article["url"]
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append(article)

        merged.sort(key=lambda a: a.get("score", 0), reverse=True)
        return merged[:max_total]

    async def close(self):
        """리소스 정리 (Tavily는 별도 정리 불필요)"""
        pass


# Singleton (lazy init)
news_service = NewsService()
