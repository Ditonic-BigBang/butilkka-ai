import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.services.model_service import model_service

logger = logging.getLogger(__name__)

# 라벨 아는 지역 3곳(청운효자동=성장, 사직동=유지, 명동=쇠퇴)으로 실측 검증된 클래스 매핑
CLASS_TREND_MAP = {0: "쇠퇴", 1: "유지", 2: "성장"}
GRADE_ORDER = ["A", "B", "C", "D", "E"]


class TrendPredictionService:
    """검증된 XGBoost 모델(decline_grade.pkl)로 다음 분기 트렌드 예측.
    32개 피처(매출QoQ/유동인구/점포수/폐업률 x 8분기) 중 가장 최근 분기 값만
    뉴스 기반 조정치를 반영해 재예측한다."""

    def __init__(self, model_dir: str = "app/models"):
        self.model_dir = Path(model_dir)
        self._scaler = None
        self._feat_cols: Optional[list[str]] = None

    def _load_scaler(self):
        if self._scaler is None:
            self._scaler = joblib.load(self.model_dir / "xgb_scaler.pkl")
        return self._scaler

    def _load_feat_cols(self) -> list[str]:
        if self._feat_cols is None:
            with open(self.model_dir / "feat_cols.json", encoding="utf-8") as f:
                self._feat_cols = json.load(f)
        return self._feat_cols

    @staticmethod
    def _adjust_latest(series: list[float], pct: float) -> list[float]:
        """가장 최근 분기(마지막 값)만 조정. 이미 확정된 과거 분기는 건드리지 않음."""
        series = list(series)
        series[-1] = series[-1] * (1 + pct)
        return series

    def predict_trend(
        self,
        sales_qoq: list[float],
        foot_traffic: list[float],
        store_count: list[float],
        closure_rate: list[float],
        adjustments: dict,
    ) -> Optional[str]:
        """8분기 이력(오래된→최근) + 최근 분기 조정치로 다음 분기 트렌드(성장/유지/쇠퇴) 예측"""
        try:
            feat_cols = self._load_feat_cols()
            if len(feat_cols) != 32:
                raise ValueError(f"feat_cols 길이 이상: {len(feat_cols)}(32 예상)")

            vector = (
                self._adjust_latest(sales_qoq, adjustments.get("sales_qoq_pct", 0) or 0)
                + self._adjust_latest(foot_traffic, adjustments.get("foot_traffic_pct", 0) or 0)
                + self._adjust_latest(store_count, adjustments.get("store_count_pct", 0) or 0)
                + self._adjust_latest(closure_rate, adjustments.get("closure_rate_pct", 0) or 0)
            )

            scaled = self._load_scaler().transform(np.array(vector).reshape(1, -1))

            model = model_service.load_model("decline_grade")
            pred_class = int(model.predict(scaled)[0])
            return CLASS_TREND_MAP.get(pred_class)
        except Exception as e:
            logger.warning(f"다음 분기 트렌드 예측 실패: {e}")
            return None

    @staticmethod
    def next_grade(current_grade: str, trend: str) -> Optional[str]:
        """현재 등급 + 예측된 트렌드 → 다음 분기 예상 등급 (한 단계 상/하, 경계값 clamp)"""
        try:
            idx = GRADE_ORDER.index(current_grade)
        except ValueError:
            return None

        if trend == "쇠퇴":
            idx = min(idx + 1, len(GRADE_ORDER) - 1)
        elif trend == "성장":
            idx = max(idx - 1, 0)
        # "유지"는 그대로

        return GRADE_ORDER[idx]


# Singleton
trend_prediction_service = TrendPredictionService()
