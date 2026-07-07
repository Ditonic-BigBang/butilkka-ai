from openai import OpenAI
from typing import Optional
import json
import logging

from app.services.news_service import news_service
from app.services.rag_service import RAGService
from app.services.cache_service import get_cache_service

logger = logging.getLogger(__name__)


class ReportService:
    """AI 리포트 생성 서비스 (ERD 기준, 3회 LLM 호출)"""

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.rag = RAGService(openai_api_key=openai_api_key)
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
        context: dict
    ) -> dict:
        """리포트 생성 파이프라인"""

        # 1. Tavily 뉴스 검색
        query = f"{district_name} {region_name} 상권"
        articles = await news_service.search_news(query=query, max_results=10)

        # 2. FAISS 인덱스 생성
        if articles:
            texts = [f"{a['title']} {a['content']}" for a in articles]
            metadatas = [{"url": a["url"], "title": a["title"]} for a in articles]
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

        # 6. 유사 사례 / 대안 지역 (all_grades 캐시)
        cache = get_cache_service()
        all_grades = cache.get_all_grades(year, quarter) or []

        similar_cases = self._find_similar_cases(
            current_grade=grade,
            current_score=score,
            decline_type=decline_type,
            all_grades=all_grades,
            exclude_code=region_code
        )

        alternative_regions = self._find_alternatives(
            current_grade=grade,
            decline_type=decline_type,
            all_grades=all_grades,
            exclude_code=region_code
        )

        # 7. 결과 조합
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
        """LLM 호출 2: 원인 + 시그널"""
        rag_context = self.rag.get_context("상권 변화 원인 시그널 트렌드", k=3)

        prompt = f"""당신은 상권 분석 전문가입니다.

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

## 요청
상권 변화의 원인과 시그널을 JSON 형식으로 응답하세요:
{{
    "causes": [
        {{"title": "원인 제목 (20자 이내)", "level": "높음/중간/낮음", "description": "설명 (50자 이내)"}},
        ...
    ],
    "signals": [
        {{"title": "시그널 제목 (20자 이내)", "description": "설명 (50자 이내)"}},
        ...
    ]
}}

- causes: 2~3개 (level은 반드시 "높음", "중간", "낮음" 중 하나)
- signals: 2~3개"""

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
        current_grade: str,
        current_score: int,
        decline_type: str,
        all_grades: list[dict],
        exclude_code: str
    ) -> list[dict]:
        """유사 사례 찾기 (같은 등급, 비슷한 점수)"""
        similar = []

        for item in all_grades:
            if item["region_code"] == exclude_code:
                continue
            if item["grade"] == current_grade:
                item_score = item.get("score", 50)
                score_diff = abs(item_score - current_score)
                if score_diff < 15:  # 점수 차이 15점 이내
                    similar.append({
                        "region_code": item["region_code"],
                        "summary": f"동일 등급({current_grade}), 유사 점수({item_score}점)",
                        "description": None,
                        "start_year": None,
                        "end_year": None,
                        "tag1": current_grade,
                        "tag2": decline_type,
                        "tag3": None,
                        "tag4": None
                    })

            if len(similar) >= 3:
                break

        return similar

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


def create_report_service(openai_api_key: str) -> ReportService:
    return ReportService(openai_api_key=openai_api_key)
