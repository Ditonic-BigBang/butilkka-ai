"""
뉴스 배치 수집 서비스
- 매일 새벽 서울 25개 구 x 9개 카테고리 뉴스 수집
- Redis에 24시간 TTL로 캐시
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from app.services.news_service import NewsService
from app.services.cache_service import get_cache_service

logger = logging.getLogger(__name__)

# 서울 25개 구
SEOUL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구"
]


class NewsBatchService:
    """뉴스 배치 수집 서비스"""

    def __init__(self, api_key: Optional[str] = None):
        self.news_service = NewsService(api_key=api_key)
        self.delay_seconds = 1.0  # API 호출 간 딜레이 (rate limit 방지)

    async def run_batch(self, districts: list[str] = None) -> dict:
        """
        전체 배치 실행

        Args:
            districts: 수집할 구 목록 (기본값: 서울 25개 구 전체)

        Returns:
            dict: 배치 실행 결과 통계
        """
        districts = districts or SEOUL_DISTRICTS
        start_time = datetime.now()

        logger.info(f"뉴스 배치 시작: {len(districts)}개 구")

        success_count = 0
        fail_count = 0
        total_articles = 0

        for district in districts:
            try:
                article_count = await self._fetch_district_news(district)
                success_count += 1
                total_articles += article_count
                logger.info(f"[{success_count}/{len(districts)}] {district}: {article_count}건 수집")
            except Exception as e:
                fail_count += 1
                logger.error(f"{district} 수집 실패: {e}")

            # 다음 구 처리 전 딜레이
            await asyncio.sleep(self.delay_seconds)

        elapsed = (datetime.now() - start_time).total_seconds()

        result = {
            "success": success_count,
            "failed": fail_count,
            "total_articles": total_articles,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": start_time.isoformat()
        }

        logger.info(f"뉴스 배치 완료: {result}")
        return result

    async def _fetch_district_news(self, district_name: str) -> int:
        """
        단일 구 뉴스 수집 (9개 카테고리)

        Returns:
            int: 수집된 총 기사 수
        """
        cache = get_cache_service()
        total_count = 0

        # 일반 상권 검색
        articles = await self.news_service.search_news(
            query=f"{district_name} 상권",
            max_results=5,
            days=90,
            category="일반"
        )
        if articles:
            cache.set_news_cache(district_name, "일반", articles)
            total_count += len(articles)

        await asyncio.sleep(0.5)  # 카테고리 간 짧은 딜레이

        # 8개 선행 신호 카테고리 검색
        for category, keyword in self.news_service.CATEGORY_QUERIES.items():
            try:
                articles = await self.news_service.search_news(
                    query=f"{district_name} {keyword}",
                    max_results=3,
                    days=90,
                    category=category
                )
                if articles:
                    cache.set_news_cache(district_name, category, articles)
                    total_count += len(articles)

                await asyncio.sleep(0.5)  # 카테고리 간 짧은 딜레이

            except Exception as e:
                logger.warning(f"{district_name}/{category} 검색 실패: {e}")

        return total_count

    async def fetch_single_district(self, district_name: str) -> dict:
        """단일 구 수동 수집 (테스트/관리용)"""
        try:
            count = await self._fetch_district_news(district_name)
            return {"district": district_name, "articles": count, "status": "success"}
        except Exception as e:
            return {"district": district_name, "error": str(e), "status": "failed"}


# Singleton (lazy init)
_batch_service: Optional[NewsBatchService] = None


def get_news_batch_service() -> NewsBatchService:
    global _batch_service
    if _batch_service is None:
        from app.core.config import get_settings
        _batch_service = NewsBatchService(api_key=get_settings().tavily_api_key)
    return _batch_service
