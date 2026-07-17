"""
관리자 API
- 뉴스 배치 수동 트리거
- 캐시 상태 조회
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = logging.getLogger(__name__)


class BatchTriggerRequest(BaseModel):
    districts: Optional[list[str]] = None  # 특정 구만 실행 (기본: 전체)


class BatchTriggerResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None


class BatchResultResponse(BaseModel):
    success: int
    failed: int
    total_articles: int
    elapsed_seconds: float
    timestamp: str


# ─────────────────────────────────────────
# 뉴스 배치 API
# ─────────────────────────────────────────

@router.post("/news-batch/trigger", response_model=BatchTriggerResponse)
async def trigger_news_batch(
    request: BatchTriggerRequest = None,
    background_tasks: BackgroundTasks = None
):
    """
    뉴스 배치 수동 실행

    - districts: 특정 구 목록 (미지정 시 서울 25개 구 전체)
    - 백그라운드에서 실행됨
    """
    from app.services.news_batch_service import get_news_batch_service

    districts = request.districts if request else None

    async def run_batch():
        try:
            batch_service = get_news_batch_service()
            result = await batch_service.run_batch(districts=districts)
            logger.info(f"수동 배치 완료: {result}")
        except Exception as e:
            logger.error(f"수동 배치 실패: {e}")

    background_tasks.add_task(run_batch)

    return BatchTriggerResponse(
        status="started",
        message=f"배치 시작됨 (구: {len(districts) if districts else 25}개)"
    )


@router.post("/news-batch/run", response_model=BatchResultResponse)
async def run_news_batch_sync(request: BatchTriggerRequest = None):
    """
    뉴스 배치 동기 실행 (완료까지 대기)

    - 테스트용, 프로덕션에서는 trigger 사용 권장
    """
    from app.services.news_batch_service import get_news_batch_service

    try:
        batch_service = get_news_batch_service()
        districts = request.districts if request else None
        result = await batch_service.run_batch(districts=districts)

        return BatchResultResponse(**result)

    except Exception as e:
        logger.error(f"배치 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/news-batch/district/{district_name}")
async def fetch_single_district(district_name: str):
    """단일 구 뉴스 수집 (테스트용)"""
    from app.services.news_batch_service import get_news_batch_service

    try:
        batch_service = get_news_batch_service()
        result = await batch_service.fetch_single_district(district_name)
        return result

    except Exception as e:
        logger.error(f"단일 구 수집 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# 캐시 상태 API
# ─────────────────────────────────────────

@router.get("/cache/news/{district_name}")
async def get_news_cache_status(district_name: str):
    """구별 뉴스 캐시 상태 조회"""
    from app.services.cache_service import get_cache_service

    try:
        cache = get_cache_service()
        articles = cache.get_all_news_cache(district_name)

        return {
            "district": district_name,
            "cached": len(articles) > 0,
            "article_count": len(articles),
            "sample": articles[:3] if articles else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/news/{district_name}")
async def delete_news_cache(district_name: str):
    """구별 뉴스 캐시 삭제"""
    from app.services.cache_service import get_cache_service

    try:
        cache = get_cache_service()
        deleted = cache.delete_news_cache(district_name)

        return {
            "district": district_name,
            "deleted_keys": deleted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
