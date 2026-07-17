"""
APScheduler 설정
- 뉴스 배치: 매일 새벽 4시 실행
"""

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _run_news_batch():
    """뉴스 배치 작업 실행"""
    from app.services.news_batch_service import get_news_batch_service

    logger.info("스케줄러: 뉴스 배치 시작")
    try:
        batch_service = get_news_batch_service()
        result = await batch_service.run_batch()
        logger.info(f"스케줄러: 뉴스 배치 완료 - {result}")
    except Exception as e:
        logger.error(f"스케줄러: 뉴스 배치 실패 - {e}")


def setup_scheduler():
    """스케줄러 작업 등록"""
    # 매일 새벽 4시 실행
    scheduler.add_job(
        _run_news_batch,
        trigger=CronTrigger(hour=4, minute=0),
        id="news_batch",
        name="Daily News Batch Collection",
        replace_existing=True
    )
    logger.info("스케줄러 작업 등록 완료: news_batch (매일 04:00)")


def start_scheduler():
    """스케줄러 시작"""
    setup_scheduler()
    scheduler.start()
    logger.info("스케줄러 시작됨")


def shutdown_scheduler():
    """스케줄러 종료"""
    scheduler.shutdown(wait=False)
    logger.info("스케줄러 종료됨")
