from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.routes import predict, report
from app.core.config import get_settings
from app.services.news_service import news_service

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await news_service.close()


app = FastAPI(
    title="Butilkka AI API",
    description="상권 분석 AI 서비스 - 모델 예측 & RAG 리포트 생성",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포시 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(predict.router, prefix="/api")
app.include_router(report.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Butilkka AI API", "docs": "/docs"}


@app.get("/health")
def health():
    from app.services.cache_service import get_cache_service
    try:
        cache = get_cache_service()
        redis_ok = cache.ping()
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected"
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
