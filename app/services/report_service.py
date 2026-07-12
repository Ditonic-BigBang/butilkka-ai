from openai import OpenAI
from typing import Optional
import json
import logging

from app.services.news_service import news_service
from app.services.rag_service import RAGService
from app.services.cache_service import get_cache_service
from app.services.case_service import get_case_service
from app.services.prediction_service import trend_prediction_service

logger = logging.getLogger(__name__)


class ReportService:
    """AI 리포트 생성 서비스 (ERD 기준, 3회 LLM 호출)"""

    def __init__(self, openai_api_key: str, chroma_db_dir: str = "chroma_db"):
        self.client = OpenAI(api_key=openai_api_key)
        self.rag = RAGService(openai_api_key=openai_api_key)
        self.case_service = get_case_service(openai_api_key=openai_api_key, persist_directory=chroma_db_dir)
        self.model = "gpt-4o-mini"

    async def generate(
        self,
        region_code: str,
        region_name: str,
        district_name: str,
        year: int,
        quarter: int,
        grade: str,           # A~E
        score: int,           # 0~100
        decline_type: str,    # 성장형/순환형/쇠퇴형/정체형
        context: dict,
        quarterly_history: dict = None,  # 8분기 이력 (있으면 다음 분기 트렌드 예측)
    ) -> dict:
        """리포트 생성 파이프라인"""

        # 1. Tavily 뉴스 검색 (구 단위, 최근 3개월, 8개 선행 신호 카테고리 확장 검색)
        articles = await news_service.search_by_categories(district_name=district_name, days=90)

        # 2. FAISS 인덱스 생성
        if articles:
            texts = [f"{a['title']} {a['content']}" for a in articles]
            metadatas = [
                {"url": a["url"], "title": a["title"], "category": a.get("category")}
                for a in articles
            ]
            self.rag.create_index(texts=texts, metadatas=metadatas)

        # 3. LLM 호출 1: 전망 + 요약
        outlook_result = self._call_outlook_summary(
            region_name=region_name,
            district_name=district_name,
            grade=grade,
            score=score,
            decline_type=decline_type,
            context=context
        )

        # 4. LLM 호출 2: 원인 + 시그널
        cause_signal_result = self._call_cause_signal(
            region_name=region_name,
            district_name=district_name,
            grade=grade,
            decline_type=decline_type,
            context=context
        )

        # 5. LLM 호출 3: 의사결정
        decision_result = self._call_decision(
            region_name=region_name,
            district_name=district_name,
            grade=grade,
            score=score,
            decline_type=decline_type,
            context=context,
            outlook=outlook_result.get("ai_outlook", "")
        )

        # 6. 유사 사례 (벡터 검색) / 대안 지역 (all_grades 캐시)
        cache = get_cache_service()
        all_grades = cache.get_all_grades(year, quarter) or []

        similar_cases = self._find_similar_cases(
            region_name=region_name,
            district_name=district_name,
            decline_type=decline_type,
            causes=cause_signal_result.get("causes", []),
            exclude_code=region_code
        )

        alternative_regions = self._find_alternatives(
            current_grade=grade,
            decline_type=decline_type,
            all_grades=all_grades,
            exclude_code=region_code
        )

        # 7. 생성된 리포트를 다음 유사사례 검색을 위해 색인 (실패해도 응답에 영향 없음)
        try:
            self.case_service.upsert_report(
                region_code=region_code,
                region_name=region_name,
                district_name=district_name,
                year=year,
                quarter=quarter,
                grade=grade,
                decline_type=decline_type,
                summary=outlook_result.get("summary", ""),
                ai_outlook=outlook_result.get("ai_outlook", ""),
                causes=cause_signal_result.get("causes", []),
            )
        except Exception as e:
            logger.warning(f"생성 리포트 벡터 색인 실패 (무시하고 계속): {e}")

        # 8. 다음 분기 예상 등급 (8분기 이력 있을 때만, 실패해도 응답엔 영향 없음)
        predicted_trend = None
        predicted_next_grade = None
        if quarterly_history:
            try:
                predicted_trend = trend_prediction_service.predict_trend(
                    sales_qoq=quarterly_history["sales_qoq"],
                    foot_traffic=quarterly_history["foot_traffic"],
                    store_count=quarterly_history["store_count"],
                    closure_rate=quarterly_history["closure_rate"],
                    adjustments=cause_signal_result.get("metric_adjustments", {}),
                )
                if predicted_trend:
                    predicted_next_grade = trend_prediction_service.next_grade(grade, predicted_trend)
            except Exception as e:
                logger.warning(f"다음 분기 예측 실패 (무시하고 계속): {e}")

        # 9. 결과 조합
        return {
            "summary": outlook_result.get("summary", ""),
            "ai_outlook": outlook_result.get("ai_outlook", ""),
            "decision_recommendation": decision_result.get("recommendation", "버티기"),
            "decision_title": decision_result.get("title", ""),
            "decision_description": decision_result.get("description", ""),
            "causes": cause_signal_result.get("causes", []),
            "signals": cause_signal_result.get("signals", []),
            "decision_reasons": decision_result.get("reasons", {}),
            "similar_cases": similar_cases,
            "alternative_regions": alternative_regions,
            "predicted_trend": predicted_trend,
            "predicted_next_grade": predicted_next_grade,
        }

    def _call_outlook_summary(
        self,
        region_name: str,
        district_name: str,
        grade: str,
        score: int,
        decline_type: str,
        context: dict
    ) -> dict:
        """LLM 호출 1: 전망 + 요약"""
        rag_context = self.rag.get_context("상권 전망 동향 변화", k=3)

        prompt = f"""당신은 상권 분석 전문가입니다.

## 분석 대상
- 지역: {district_name} {region_name}
- 등급: {grade} (점수: {score}점)
- 유형: {decline_type}
- 매출 변화율: {context.get('sales_delta', 'N/A')}%
- 유동인구 변화율: {context.get('foot_traffic_delta', 'N/A')}%
- 폐업률: {context.get('closure_rate', 'N/A')}%

## 관련 뉴스/정보
{rag_context if rag_context else "관련 정보 없음"}

## 요청
위 정보를 바탕으로 JSON 형식으로 응답하세요:
{{
    "summary": "한 줄 요약 (50자 이내)",
    "ai_outlook": "AI 종합 전망 (5~6줄, 구체적인 이유 포함. 예: 재개발 예정, 대형 건물 입주, 인구 감소 등)"
}}"""

        return self._call_llm_json(prompt)

    def _call_cause_signal(
        self,
        region_name: str,
        district_name: str,
        grade: str,
        decline_type: str,
        context: dict
    ) -> dict:
        """LLM 호출 2: 선행 신호 + 원인 (다음 분기 예측용 — 과거 분기를 사후 설명하는 게 아님)"""
        rag_context = self.rag.get_context("상권 변화 원인 시그널 트렌드", k=6)

        category_checklist = "\n".join(f"- {c}" for c in news_service.CATEGORY_QUERIES)

        prompt = f"""당신은 상권 분석 전문가입니다. 이 리포트는 다음 분기 상권 전망을 예측하기 위한
선행 분석입니다. 이미 확정된 과거 분기를 사후적으로 설명하는 것이 아니라, 지금 막 나타나고
있는 조짐이 다음 분기에 어떤 영향을 줄지 파악하는 것이 목적입니다.

## 분석 대상
- 지역: {district_name} {region_name}
- 등급: {grade}
- 유형: {decline_type}
- 매출 변화율: {context.get('sales_delta', 'N/A')}%
- 유동인구 변화율: {context.get('foot_traffic_delta', 'N/A')}%
- 점포수 변화율: {context.get('store_count_delta', 'N/A')}%
- 폐업률: {context.get('closure_rate', 'N/A')}%
- 공실률: {context.get('vacancy_rate', 'N/A')}%

## 관련 뉴스/정보
{rag_context if rag_context else "관련 정보 없음"}

## 참고할 선행 신호 축 (분류 카테고리일 뿐, 이 이름을 그대로 title에 쓰지 말 것)
{category_checklist}

## 요청
아래 순서로 판단해서 JSON으로만 응답하세요:
1. 먼저 위 뉴스에서 다음 분기에 영향을 줄 만한 구체적 사실(선행 신호)을 뽑는다.
2. 그 신호들을 근거로, 어떤 구체적 요인이 얼마나 심각하게 상권에 영향을 미치는지(원인) 해석한다.

**중요**: 이 상권의 현재 유형은 "{decline_type}"이다. signals/causes는 반드시 이 유형을
설명하거나 다음 분기에 강화할 방향으로만 작성한다. 뉴스가 반대 방향(예: 쇠퇴형인데 상권
활성화 정책·개발 호재 뉴스만 있는 경우)이면 그 뉴스를 원인으로 삼지 말고, 대신 주어진
지표(매출/유동인구/폐업률 등) 변화를 근거로 signals/causes를 구성한다.

{{
    "signals": [
        {{"title": "선행 신호 (뉴스 기반 짧은 문구, 25자 이내, 완전한 문장 아님)"}},
        ...
    ],
    "causes": [
        {{"title": "구체적 원인 요인 이름 (25자 이내, 완전한 문장 아님)", "level": "높음/중간/낮음"}},
        ...
    ],
    "metric_adjustments": {{
        "sales_qoq_pct": 0.0,
        "foot_traffic_pct": 0.0,
        "store_count_pct": 0.0,
        "closure_rate_pct": 0.0
    }}
}}

- signals: 2~3개. 뉴스에서 실제로 확인된 사실만 (관련 뉴스 없으면 지표 기반 조짐으로 대체 가능)
- causes: 2~3개. level은 반드시 "높음", "중간", "낮음" 중 하나
- causes의 title은 "재개발/도시계획", "부동산/임대료" 같은 위 카테고리 이름 자체를 쓰면 안 됨.
  반드시 신호에서 확인된 구체적 내용을 담은 이름으로 작성할 것
  (예: 카테고리가 "부동산/임대료"라면 title은 "임대료 급등 및 젠트리피케이션"처럼 구체화)
- title은 UI에 한 줄로 표시되므로 설명 문장이 아니라 짧은 명사구로 작성
- metric_adjustments: signals/causes에서 확인된 조짐이 **다음 분기** 각 지표에 미칠 영향을
  -0.3~0.3 범위의 비율로 추정 (악화면 음수, 개선이면 양수). 관련된 신호가 없는 지표는 0.
  확대해석 금지 — 명확한 근거가 있는 지표만 0이 아닌 값을 준다."""

        return self._call_llm_json(prompt)

    def _call_decision(
        self,
        region_name: str,
        district_name: str,
        grade: str,
        score: int,
        decline_type: str,
        context: dict,
        outlook: str
    ) -> dict:
        """LLM 호출 3: 의사결정"""
        rag_context = self.rag.get_context("상권 창업 투자 위험 기회", k=3)

        prompt = f"""당신은 상권 분석 전문가입니다. 소상공인의 의사결정을 도와주세요.

## 분석 대상
- 지역: {district_name} {region_name}
- 등급: {grade} (점수: {score}점)
- 유형: {decline_type}
- 전망: {outlook}

## 지표
- 매출 변화율: {context.get('sales_delta', 'N/A')}%
- 유동인구 변화율: {context.get('foot_traffic_delta', 'N/A')}%
- 폐업률: {context.get('closure_rate', 'N/A')}%
- 주요 연령대: {context.get('top_age_group', 'N/A')}
- 주요 성별: {context.get('top_gender', 'N/A')}

## 관련 뉴스/정보
{rag_context if rag_context else "관련 정보 없음"}

## 요청
의사결정 권고를 JSON 형식으로 응답하세요:
{{
    "recommendation": "버티기 또는 이동",
    "title": "권고 제목 (예: '현 위치 유지 권고', '이전 검토 필요')",
    "description": "권고 설명 (5~6줄, 구체적 근거 포함)",
    "reasons": {{
        "reason_1": "첫 번째 근거 (30자 이내)",
        "reason_2": "두 번째 근거 (30자 이내)",
        "reason_3": "세 번째 근거 또는 null (30자 이내)"
    }}
}}

- recommendation: 반드시 "버티기" 또는 "이동" 중 하나
- 버티기: 현재 위치에서 영업 지속 권고
- 이동: 다른 상권으로 이전 검토 권고"""

        return self._call_llm_json(prompt)

    def _call_llm_json(self, prompt: str) -> dict:
        """LLM 호출 (JSON 응답)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            return {}

    def _find_similar_cases(
        self,
        region_name: str,
        district_name: str,
        decline_type: str,
        causes: list[dict],
        exclude_code: str
    ) -> list[dict]:
        """유사 사례 찾기 (벡터 검색: 큐레이션 사례 + 과거 생성 리포트)"""
        cause_titles = " ".join(c.get("title", "") for c in causes)
        query = f"{district_name} {region_name} {decline_type} {cause_titles}".strip()

        try:
            results = self.case_service.search_similar(query=query, k=3, exclude_region_code=exclude_code)
        except Exception as e:
            logger.warning(f"유사 사례 벡터 검색 실패, 빈 목록 반환: {e}")
            return []

        return [
            {
                "region_code": meta["region_code"],
                "summary": meta.get("summary", ""),
                "description": meta.get("description"),
                "start_year": meta.get("start_year"),
                "end_year": meta.get("end_year"),
                "tag1": meta.get("tag1"),
                "tag2": meta.get("tag2"),
                "tag3": meta.get("tag3"),
                "tag4": meta.get("tag4"),
            }
            for meta in results
        ]

    def _find_alternatives(
        self,
        current_grade: str,
        decline_type: str,
        all_grades: list[dict],
        exclude_code: str
    ) -> list[dict]:
        """대안 지역 찾기 (더 좋은 등급)"""
        grade_order = ["A", "B", "C", "D", "E"]

        try:
            current_idx = grade_order.index(current_grade)
        except ValueError:
            return []

        alternatives = []

        for item in all_grades:
            if item["region_code"] == exclude_code:
                continue

            item_grade = item.get("grade", "C")
            try:
                item_idx = grade_order.index(item_grade)
            except ValueError:
                continue

            if item_idx < current_idx:  # 더 좋은 등급 (A < B < C < D < E)
                item_score = item.get("score", 50)
                alternatives.append({
                    "region_code": item["region_code"],
                    "reason": f"{item_grade}등급 상권 (현재보다 양호)",
                    "stat": f"점수 {item_score}점"
                })

            if len(alternatives) >= 3:
                break

        return alternatives


def create_report_service(openai_api_key: str, chroma_db_dir: str = "chroma_db") -> ReportService:
    return ReportService(openai_api_key=openai_api_key, chroma_db_dir=chroma_db_dir)
