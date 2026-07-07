from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_service import model_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


class PredictRequest(BaseModel):
    model_name: str
    data: list[dict]


class PredictResponse(BaseModel):
    predictions: list
    model_name: str


@router.get("/models")
def list_models():
    """사용 가능한 모델 목록"""
    models = model_service.list_models()
    return {"models": models}


@router.post("/", response_model=PredictResponse)
def predict(request: PredictRequest):
    """모델 예측"""
    try:
        predictions = model_service.predict(
            model_name=request.model_name,
            data=request.data
        )
        return PredictResponse(
            predictions=predictions,
            model_name=request.model_name
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
