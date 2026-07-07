import redis
import json
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class CacheService:
    """Redis 캐시 서비스"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 86400  # 1일

    # ─────────────────────────────────────────
    # 등급 캐시
    # ─────────────────────────────────────────

    def set_grade(
        self,
        region_code: str,
        year: int,
        quarter: int,
        grade: str,
        score: float
    ) -> None:
        """등급 캐시 저장"""
        key = f"grade:{region_code}:{year}:{quarter}"
        value = json.dumps({"grade": grade, "score": score})
        self.redis.setex(key, self.default_ttl, value)

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
        """전체 등급 리스트 캐시 (similar_cases용)"""
        key = f"all_grades:{year}:{quarter}"
        value = json.dumps(grades)
        self.redis.setex(key, self.default_ttl, value)

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
    # 뉴스/임베딩 캐시
    # ─────────────────────────────────────────

    def set_news_cache(
        self,
        query: str,
        articles: list[dict],
        ttl: int = 3600  # 1시간
    ) -> None:
        """뉴스 검색 결과 캐시"""
        key = f"news:{query}"
        value = json.dumps(articles)
        self.redis.setex(key, ttl, value)

    def get_news_cache(self, query: str) -> Optional[list[dict]]:
        """뉴스 검색 결과 조회"""
        key = f"news:{query}"
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    # ─────────────────────────────────────────
    # 범용 메서드
    # ─────────────────────────────────────────

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """범용 캐시 저장"""
        serialized = json.dumps(value)
        if ttl:
            self.redis.setex(key, ttl, serialized)
        else:
            self.redis.setex(key, self.default_ttl, serialized)

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
