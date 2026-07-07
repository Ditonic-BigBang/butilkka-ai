# Butilkka AI

상권 분석 AI 서비스 - 모델 예측 & RAG 기반 리포트 생성

## 구조

```
butilkka-ai/
├── main.py                 # FastAPI 엔트리포인트
├── app/
│   ├── api/routes/
│   │   ├── predict.py      # 모델 예측 API
│   │   └── report.py       # AI 리포트 API
│   ├── core/
│   │   └── config.py       # 설정
│   ├── services/
│   │   ├── model_service.py    # PKL 모델 로드/예측
│   │   ├── news_service.py     # 뉴스 API 크롤링
│   │   └── rag_service.py      # RAG 벡터 임베딩
│   └── models/             # PKL 모델 파일 위치
├── requirements.txt
└── .env
```

## 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

## 실행

```bash
python main.py
# 또는
uvicorn main:app --reload --port 8000
```

## API 문서

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 모델 예측
- `GET /api/predict/models` - 사용 가능한 모델 목록
- `POST /api/predict` - 모델 예측

### AI 리포트
- `POST /api/report/news/search` - 뉴스 검색
- `POST /api/report/embed` - 문서 벡터 임베딩
- `POST /api/report/search` - 유사 문서 검색
- `POST /api/report/context` - RAG 컨텍스트 생성

## 모델 추가

`app/models/` 폴더에 `.pkl` 파일을 추가하면 자동으로 인식됩니다.

```python
# 모델 저장 예시
import joblib
joblib.dump(model, "app/models/my_model.pkl")
```
