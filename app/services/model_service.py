import joblib
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ModelService:
    """PKL 모델 로드 및 예측 서비스"""

    def __init__(self, model_dir: str = "app/models"):
        self.model_dir = Path(model_dir)
        self._models: dict[str, Any] = {}

    def load_model(self, model_name: str) -> Any:
        """모델 로드 (캐싱)"""
        if model_name in self._models:
            return self._models[model_name]

        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model: {model_path}")
        model = joblib.load(model_path)
        self._models[model_name] = model
        return model

    def predict(self, model_name: str, data: list[dict]) -> list:
        """모델 예측"""
        import pandas as pd

        model = self.load_model(model_name)
        df = pd.DataFrame(data)
        predictions = model.predict(df)
        return predictions.tolist()

    def list_models(self) -> list[str]:
        """사용 가능한 모델 목록"""
        if not self.model_dir.exists():
            return []
        return [f.stem for f in self.model_dir.glob("*.pkl")]


# Singleton
model_service = ModelService()
