from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    news_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    model_dir: str = "app/models"
    chroma_db_dir: str = "chroma_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
