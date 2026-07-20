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


def _normalize_region_code(region_code: Optional[str]) -> str:
    if not region_code:
        return ""
    code = str(region_code).strip()
    if len(code) == 10 and code.isdigit():
        return code[:5]
    return code


def _normalize_region_name(region_name: Optional[str], district_name: Optional[str]) -> str:
    return (region_name or district_name or "").strip()


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
        normalized_region_code = _normalize_region_code(region_code)
        normalized_region_name = _normalize_region_name(region_name, district_name)
        normalized_district_name = normalized_region_name or district_name or region_name or ""

        # 1. Tavily 뉴스 검색 (구 단위, 최근 3개월, 8개 선행 신호 카테고리 확장 검색)
        articles = await news_service.search_by_categories(district_name=normalized_district_name, days=90)

        # 2. FAISS 인덱스 생성
        if articles:
            texts = [f"{a['title']} {a['content']}" for a in articles]
            metadatas = [
                {"url": a["url"], "title": a["title"], "category": a.get("category")}
                for a in articles
            ]
            self.rag.create_index(texts=texts, metadatas=metadatas)

        # 3. LLM 호출 1: 전망 + 요약
        normalized_context = {
            **context,
            "top_age_group": None,
            "top_gender": None,
        }

        outlook_result = self._call_outlook_summary(
            region_name=normalized_region_name,
            district_name=normalized_district_name,
            grade=grade,
            score=score,
            decline_type=decline_type,
            context=normalized_context
        )

        # 4. LLM 호출 2: 원인 + 시그널
        cause_signal_result = self._call_cause_signal(
            region_name=normalized_region_name,
            district_name=normalized_district_name,
            grade=grade,
            decline_type=decline_type,
            context=normalized_context
        )

        # 5. LLM 호출 3: 의사결정
        decision_result = self._call_decision(
            region_name=normalized_region_name,
            district_name=normalized_district_name,
            grade=grade,
            score=score,
            decline_type=decline_type,
            context=normalized_context,
            outlook=outlook_result.get("ai_outlook", "")
        )

        # 6. 유사 사례 (벡터 검색) / 대안 지역 (all_grades 캐시)
        cache = get_cache_service()
        all_grades = cache.get_all_grades(year, quarter) or []

        similar_cases = self._find_similar_cases(
            region_name=normalized_region_name,
            district_name=normalized_district_name,
            decline_type=decline_type,
            causes=cause_signal_result.get("causes", []),
            exclude_code=normalized_region_code,
            signals=cause_signal_result.get("signals", [])
        )

        alternative_regions = self._find_alternatives(
            current_region_name=normalized_region_name,
            current_grade=grade,
            current_score=score,
            decline_type=decline_type,
            all_grades=all_grades,
            exclude_code=normalized_region_code,
            year=year,
            quarter=quarter
        )

        # 7. 다음 분기 예상 등급 (8분기 이력 있을 때만, 실패해도 응답엔 영향 없음)
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

        # 8. AI 추천 카드 생성
        recommendation = decision_result.get("recommendation", "버티기")
        ai_recommendation = {
            "badge_type": "AI 추천",
            "title": "현 위치 유지를 추천드려요" if recommendation == "버티기" else "이동을 추천드려요",
            "reason_title": decision_result.get("title", ""),
            "reason_detail": decision_result.get("description", "")
        }

        # 9. 결과 조합
        return {
            "summary": outlook_result.get("summary", ""),
            "ai_outlook": outlook_result.get("ai_outlook", ""),
            "decision_recommendation": recommendation,
            "decision_title": decision_result.get("title", ""),
            "decision_description": decision_result.get("description", ""),
            "causes": cause_signal_result.get("causes", []),
            "signals": cause_signal_result.get("signals", []),
            "decision_reasons": decision_result.get("reasons", {}),
            "similar_cases": similar_cases,
            "alternative_regions": alternative_regions,
            "ai_recommendation": ai_recommendation,
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

        # 지역명 표시: district_name과 region_name이 같으면 한 번만
        display_region = district_name if district_name == region_name or not region_name else f"{district_name} {region_name}"

        prompt = f"""당신은 상권 분석 전문가입니다.

## 분석 대상
- 지역: {display_region}
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
}}

## 중요 제약사항
- ai_outlook에서 지역명을 연달아 두 번 반복하지 마세요 (예: "강남구 강남구", "명동 명동" 금지)
- 지역명은 문장 내에서 한 번만 언급하세요
- 구체적인 수치(%, 숫자)는 포함하지 마세요. 대신 "증가", "감소", "상승세", "하락세" 등 정성적 표현을 사용하세요"""

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
2. 그 신호 하나하나에 대해, 그 신호가 상권에 어떤 요인으로 얼마나 심각하게 영향을 미치는지(원인) 해석한다.
   즉 신호와 원인은 반드시 1:1로 짝지어진다 — signal_causes 배열의 각 항목이 "이 신호 → 이 원인"이다.

**중요**: 이 상권의 현재 유형은 "{decline_type}"이다. signal/cause는 반드시 이 유형을
설명하거나 다음 분기에 강화할 방향으로만 작성한다. 뉴스가 반대 방향(예: 쇠퇴형인데 상권
활성화 정책·개발 호재 뉴스만 있는 경우)이면 그 뉴스를 원인으로 삼지 말고, 대신 주어진
지표(매출/유동인구/폐업률 등) 변화를 근거로 signal_causes를 구성한다.

{{
    "signal_causes": [
        {{
            "signal": "선행 신호 (뉴스 기반 짧은 문구, 25자 이내, 완전한 문장 아님)",
            "cause": "그 신호에 따른 구체적 원인 요인 이름 (25자 이내, 완전한 문장 아님)",
            "level": "높음/중간/낮음"
        }},
        ... (정확히 3개)
    ],
    "metric_adjustments": {{
        "sales_qoq_pct": 0.0,
        "foot_traffic_pct": 0.0,
        "store_count_pct": 0.0,
        "closure_rate_pct": 0.0
    }}
}}

- signal_causes: **반드시 정확히 3개**. 뉴스에서 실제로 확인된 사실 기반으로 (관련 뉴스가
  3개보다 적으면 나머지는 지표 기반 조짐으로 채워서라도 3개를 맞춘다)
- level은 반드시 "높음", "중간", "낮음" 중 하나
- cause는 "재개발/도시계획", "부동산/임대료" 같은 카테고리 이름 자체를 쓰면 안 됨.
  반드시 그 signal에서 확인된 구체적 내용을 담은 이름으로 작성할 것
  (예: signal이 임대료 관련이면 cause는 "임대료 급등 및 젠트리피케이션"처럼 구체화)
- signal/cause 모두 UI에 한 줄로 표시되므로 설명 문장이 아니라 짧은 명사구로 작성
- metric_adjustments: signal_causes에서 확인된 조짐이 **다음 분기** 각 지표에 미칠 영향을
  -0.3~0.3 범위의 비율로 추정 (악화면 음수, 개선이면 양수). 관련된 신호가 없는 지표는 0.
  확대해석 금지 — 명확한 근거가 있는 지표만 0이 아닌 값을 준다."""

        result = self._call_llm_json(prompt)
        signal_causes = result.get("signal_causes", [])
        result["signals"] = [
            {"title": sc.get("signal", ""), "description": ""}
            for sc in signal_causes
        ]
        result["causes"] = [
            {"title": sc.get("cause", ""), "level": sc.get("level", "중간"), "description": ""}
            for sc in signal_causes
        ]
        return result

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
    "description": "권고 설명 (2~3문장, 구체적 근거 포함)",
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
        exclude_code: str,
        signals: list[dict] = None
    ) -> list[dict]:
        """유사 사례 찾기 (벡터 검색: 큐레이션 사례 + 과거 생성 리포트)

        지역명/유형이 아니라 '원인'과 '선행신호'가 비슷한 사례를 찾는 게 목적이라,
        쿼리를 원인(causes)과 신호(signals) 위주로 구성한다.
        지역명을 섞으면 엉뚱하게 지명 자체로 매칭되는 경우가 생겨서 제외.
        """
        cause_titles = " ".join(c.get("title", "") for c in causes)
        signal_titles = " ".join(s.get("title", "") for s in (signals or []))
        # 원인과 신호를 함께 사용하여 더 정확한 매칭
        query = f"{cause_titles} {signal_titles}".strip()
        if not query:
            query = f"{decline_type}"  # 지역명 제외, 유형만 사용

        try:
            results = self.case_service.search_similar(query=query, k=5, exclude_region_code=exclude_code)
        except Exception as e:
            logger.warning(f"유사 사례 벡터 검색 실패, 빈 목록 반환: {e}")
            return []

        return [
            {
                "region_code": _normalize_region_code(meta.get("region_code") or meta.get("district_code") or exclude_code),
                "region_name": (meta.get("district_name") or meta.get("region_name") or district_name or "").strip(),
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
        current_region_name: str,
        current_grade: str,
        current_score: int,
        decline_type: str,
        all_grades: list[dict],
        exclude_code: str,
        year: int,
        quarter: int
    ) -> list[dict]:
        """대안 지역(= 상승세인 대안 상권 추천) 찾기.
        더 좋은 등급 우선, 부족하면 점수 높은 순으로 최대 3개까지 채우고,
        추천 사유는 LLM으로 자연스럽게 작성."""
        grade_order = ["A", "B", "C", "D", "E"]

        try:
            current_idx = grade_order.index(current_grade)
        except ValueError:
            current_idx = None

        candidates = [item for item in all_grades if item["region_code"] != exclude_code]

        def grade_idx(item: dict) -> int:
            try:
                return grade_order.index(item.get("grade", "C"))
            except ValueError:
                return len(grade_order)

        # 더 좋은 등급 우선, 그다음 점수 높은 순
        if current_idx is not None:
            better = sorted(
                (i for i in candidates if grade_idx(i) < current_idx),
                key=lambda i: -i.get("score", 0)
            )
        else:
            better = []

        rest = sorted(
            (i for i in candidates if i not in better),
            key=lambda i: -i.get("score", 0)
        )

        picked = (better + rest)[:3]  # 최대 3개

        ai_messages = self._call_alternative_messages(
            current_region_name=current_region_name,
            current_grade=current_grade,
            current_score=current_score,
            decline_type=decline_type,
            candidates=picked,
            year=year,
            quarter=quarter
        )

        results = []
        for idx, item in enumerate(picked):
            region_code = _normalize_region_code(item.get("region_code"))
            region_name = (item.get("region_name") or item.get("district_name") or "").strip()
            results.append({
                "rank": idx + 1,
                "region_code": region_code,
                "dong_name": region_name,  # 현재는 구 이름 사용
                "ai_message": ai_messages.get(region_code) or f"{region_name}은 {item.get('grade', 'C')}등급 상권으로 현재보다 양호한 상태예요."
            })
        return results

    def _call_alternative_messages(
        self,
        current_region_name: str,
        current_grade: str,
        current_score: int,
        decline_type: str,
        candidates: list[dict],
        year: int,
        quarter: int
    ) -> dict[str, str]:
        """대안 상권별 AI 메시지를 LLM으로 생성."""
        if not candidates:
            return {}

        candidates_text = "\n".join(
            f"- region_code={_normalize_region_code(item.get('region_code'))}, 지역명={(item.get('region_name') or item.get('district_name') or '').strip()}, "
            f"등급={item.get('grade', 'C')}, 유형={item.get('decline_type', '정체')}"
            for item in candidates
        )

        prompt = f"""당신은 상권 분석 전문가입니다. 현재 상권이 부진해 이전을 고민 중인 소상공인에게
대안 상권 후보를 추천하려 합니다.

## 현재 상권
- 지역: {current_region_name}
- 등급: {current_grade}
- 유형: {decline_type}

## 대안 후보
{candidates_text}

## 요청
각 후보에 대해 소상공인에게 보여줄 추천 메시지를 작성하세요.
- 자연스러운 문장으로 작성 (예: "최근 유동인구가 꾸준히 증가하고 있어요.")
- 등급과 유형 정보를 활용해 긍정적인 메시지로 작성
- 구체적인 수치(%, 점수)는 포함하지 마세요
- 각 메시지는 40~60자 내외

JSON으로만 응답:
{{
    "messages": {{
        "<region_code>": "AI 추천 메시지"
    }}
}}"""

        result = self._call_llm_json(prompt)
        return result.get("messages", {}) or {}


def create_report_service(openai_api_key: str, chroma_db_dir: str = "chroma_db") -> ReportService:
    return ReportService(openai_api_key=openai_api_key, chroma_db_dir=chroma_db_dir)
