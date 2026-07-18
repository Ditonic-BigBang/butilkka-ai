import redis
import json
import hashlib
import unicodedata
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class CacheService:
    """Redis 캐시 서비스"""

    # TTL 정책
    TTL_PERMANENT = None      # 영구 (grade, all_grades)
    TTL_NEWS = 21600          # 6시간 (뉴스 임베딩 플래그)
    TTL_NEWS_CACHE = 86400    # 24시간 (뉴스 캐시)

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    # ─────────────────────────────────────────
    # 등급 캐시 (영구)
    # ─────────────────────────────────────────

    def set_grade(
        self,
        region_code: str,
        year: int,
        quarter: int,
        grade: str,
        score: int,
        decline_type: str
    ) -> None:
        """등급 캐시 저장 (영구)"""
        key = f"grade:{region_code}:{year}:{quarter}"
        value = json.dumps({
            "grade": grade,
            "score": score,
            "decline_type": decline_type
        })
        self.redis.set(key, value)  # TTL 없음

    def get_grade(
        self,
        region_code: str,
        year: int,
        quarter: int
    ) -> Optional[dict]:
        """등급 캐시 조회"""
        key = f"grade:{region_code}:{year}:{quarter}"
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def set_all_grades(
        self,
        year: int,
        quarter: int,
        grades: list[dict]
    ) -> None:
        """전체 등급 리스트 캐시 (영구)"""
        key = f"all_grades:{year}:{quarter}"
        value = json.dumps(grades)
        self.redis.set(key, value)  # TTL 없음

    def get_all_grades(
        self,
        year: int,
        quarter: int
    ) -> Optional[list[dict]]:
        """전체 등급 리스트 조회"""
        key = f"all_grades:{year}:{quarter}"
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    # ─────────────────────────────────────────
    # 뉴스 캐시 (6시간 TTL)
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize_query(query: str) -> str:
        """쿼리 정규화 (공백, 한글 NFD 처리)"""
        norm = " ".join(query.strip().split())  # 공백 정규화
        norm = unicodedata.normalize('NFC', norm)  # 한글 NFD → NFC
        return hashlib.md5(norm.encode()).hexdigest()

    def set_news_embedded(self, query: str) -> None:
        """뉴스 임베딩 완료 플래그 (실제 벡터는 ChromaDB)"""
        key = f"news_embedded:{self._normalize_query(query)}"
        self.redis.setex(key, self.TTL_NEWS, "1")

    def is_news_embedded(self, query: str) -> bool:
        """뉴스 임베딩 여부 확인"""
        key = f"news_embedded:{self._normalize_query(query)}"
        return self.redis.exists(key) == 1

    # ─────────────────────────────────────────
    # 뉴스 배치 캐시 (24시간 TTL)
    # ─────────────────────────────────────────

    def set_news_cache(
        self,
        district_name: str,
        category: str,
        articles: list[dict]
    ) -> None:
        """뉴스 검색 결과 캐시 저장 (24시간 TTL)"""
        key = f"news_cache:{district_name}:{category}"
        self.redis.setex(key, self.TTL_NEWS_CACHE, json.dumps(articles))

    def get_news_cache(
        self,
        district_name: str,
        category: str
    ) -> Optional[list[dict]]:
        """뉴스 캐시 조회"""
        key = f"news_cache:{district_name}:{category}"
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def get_all_news_cache(self, district_name: str) -> list[dict]:
        """구 단위 전체 카테고리 뉴스 캐시 조회 및 병합"""
        pattern = f"news_cache:{district_name}:*"
        keys = list(self.redis.scan_iter(match=pattern))

        if not keys:
            return []

        seen_urls = set()
        merged = []

        for key in keys:
            value = self.redis.get(key)
            if value:
                articles = json.loads(value)
                for article in articles:
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        merged.append(article)

        # 스코어 높은 순 정렬
        merged.sort(key=lambda a: a.get("score", 0), reverse=True)
        return merged

    def delete_news_cache(self, district_name: str) -> int:
        """구 단위 뉴스 캐시 삭제"""
        pattern = f"news_cache:{district_name}:*"
        keys = list(self.redis.scan_iter(match=pattern))
        if keys:
            return self.redis.delete(*keys)
        return 0

    def has_any_news_cache(self) -> bool:
        """뉴스 캐시가 하나라도 있는지 확인"""
        pattern = "news_cache:*"
        for _ in self.redis.scan_iter(match=pattern, count=1):
            return True
        return False

    # ─────────────────────────────────────────
    # 범용 메서드
    # ─────────────────────────────────────────

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """범용 캐시 저장"""
        serialized = json.dumps(value)
        if ttl:
            self.redis.setex(key, ttl, serialized)
        else:
            self.redis.set(key, serialized)  # 영구

    def get(self, key: str) -> Optional[Any]:
        """범용 캐시 조회"""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def delete(self, key: str) -> None:
        """캐시 삭제"""
        self.redis.delete(key)

    def ping(self) -> bool:
        """연결 확인"""
        try:
            return self.redis.ping()
        except Exception:
            return False


# Singleton (lazy init)
_cache_service: Optional[CacheService] = None


def get_cache_service(redis_url: str = None) -> CacheService:
    global _cache_service
    if _cache_service is None:
        from app.core.config import get_settings
        url = redis_url or get_settings().redis_url
        _cache_service = CacheService(redis_url=url)
    return _cache_service
