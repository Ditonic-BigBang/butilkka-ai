"""과거 뉴스에서 실제로 확정된 상권 흥망 사례를 발굴해 벡터DB에 저장.

수동 실행: python -m app.scripts.discover_cases_from_news
같은 사례가 재발견되면 case_id(지역명+기간 기반)가 같아서 덮어써짐 (idempotent).
"""
import asyncio
import json
import logging
import re

from openai import OpenAI

from app.core.config import get_settings
from app.services.news_service import news_service
from app.services.case_service import get_case_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CASE_DISCOVERY_QUERIES = [
    "상권 쇠퇴 사례",
    "상권 부활 사례",
    "젠트리피케이션 사례",
    "상권 몰락 원인",
    "골목상권 재생 사례",
    "상권 살아난 비결",
]


def slugify(region_name: str, start_year: int, end_year: int) -> str:
    slug = re.sub(r"[^0-9a-zA-Z가-힣]", "", region_name)
    return f"{slug}_{start_year}_{end_year}"


def extract_cases(client: OpenAI, articles: list[dict]) -> list[dict]:
    """뉴스 기사들에서 확정된 결과가 있는 상권 사례만 구조화 추출"""
    if not articles:
        return []

    articles_text = "\n\n".join(
        f"[{i + 1}] {a['title']}\n{a['content']}" for i, a in enumerate(articles)
    )

    prompt = f"""아래는 상권(상업 지구) 관련 뉴스 기사 모음입니다.

{articles_text}

## 요청
이 기사들에서 **실제로 확정된 결과가 있는** 상권 흥망 사례를 추출하세요.
- 반드시 "이미 일어난 일"만 포함 (예: "~했다", "~됐다"). "~될 것이다", "~전망된다" 같은
  예측/전망성 기사는 제외.
- 같은 사례가 여러 기사에 걸쳐 나오면 하나로 합쳐서 추출.
- 관련된 확정 사례가 없으면 빈 배열 반환.

JSON 형식으로만 응답:
{{
    "cases": [
        {{
            "region_name": "상권/지역 이름",
            "district_name": "소속 구 (모르면 빈 문자열)",
            "start_year": 2015,
            "end_year": 2019,
            "decline_type": "쇠퇴형 또는 성장형",
            "summary": "한 줄 요약 (50자 이내)",
            "description": "상세 설명 (아래 5문장 구조로 작성)",
            "tags": ["키워드1", "키워드2", "키워드3", "키워드4"]
        }}
    ]
}}

description은 아래 5문장 구조로 작성 (각 항목 한 문장씩, 총 5문장, 전부 과거형/확정형으로
서술 — 이미 끝난 사례이므로 "예상된다", "될 것이다" 같은 예측 표현은 쓰지 않는다):
1) 어떤 상권에서 무슨 상황이었는지 (개요)
2) 그 상황의 구체적 원인
3) 시간에 따라 어떻게 전개됐는지
4) 실제로 어떤 결과가 나왔는지 (확정된 결과)
5) 그 결과의 규모나 의미 (수치·사회적 의미 등, 기사에 있으면 포함)
기사에 정보가 부족해 일부 문장을 채우기 어려우면 자연스럽게 간략히 서술하되, 5문장 구조
자체는 유지한다."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("cases", [])
    except Exception as e:
        logger.error(f"사례 추출 실패: {e}")
        return []


def enrich_with_stats(client: OpenAI, case: dict, articles: list[dict]) -> str:
    """추가 검색된 신뢰 가능한 기사로 description 5번 문장(수치·규모)을 보강.
    자료에 실제 수치가 없으면 지어내지 않고 기존 설명을 그대로 유지."""
    if not articles:
        return case["description"]

    articles_text = "\n\n".join(
        f"[{i + 1}] {a['title']}\n{a['content']}" for i, a in enumerate(articles)
    )

    prompt = f"""기존에 작성된 상권 사례 설명입니다:

{case['description']}

아래는 이 사례의 수치·통계를 확인하기 위해 추가로 검색한 신뢰 가능한 언론사 기사입니다:

{articles_text}

## 요청
위 설명의 5번째 문장(규모나 의미)에 구체적 수치가 없거나 부족하다면, 위 추가 자료에서
**실제로 확인되는** 수치(매출, 폐업률, 공실률, 점포수, 임대료 등)가 있는 경우에만 반영해서
전체 설명(5문장)을 다시 작성하세요.

**중요**: 추가 자료에 수치가 없으면 절대 지어내지 말고, 원래 설명을 그대로 반환하세요.
확실하지 않은 수치를 추측해서 넣는 것보다 수치 없이 정성적으로 서술하는 게 낫습니다.

JSON으로만 응답: {{"description": "..."}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("description") or case["description"]
    except Exception as e:
        logger.warning(f"수치 보강 실패, 기존 설명 유지: {e}")
        return case["description"]


async def discover_all(client: OpenAI) -> list[dict]:
    all_cases = []
    for query in CASE_DISCOVERY_QUERIES:
        articles = await news_service.search_news(query=query, max_results=10, days=3650)
        logger.info(f"'{query}' 검색 결과: {len(articles)}건")
        if not articles:
            continue

        cases = extract_cases(client, articles)
        logger.info(f"  → 추출된 사례: {len(cases)}건")
        all_cases.extend(cases)

    # 사례별로 수치 보강용 추가 검색 (news_service는 신뢰 가능한 언론사 도메인만 검색함)
    for case in all_cases:
        if not case.get("region_name"):
            continue
        followup_query = f"{case['region_name']} 상권 매출 통계"
        followup_articles = await news_service.search_news(query=followup_query, max_results=5, days=3650)
        case["description"] = enrich_with_stats(client, case, followup_articles)

    return all_cases


def main():
    settings = get_settings()
    case_service = get_case_service(
        openai_api_key=settings.openai_api_key,
        persist_directory=settings.chroma_db_dir,
    )
    client = OpenAI(api_key=settings.openai_api_key)

    cases = asyncio.run(discover_all(client))

    saved = 0
    for case in cases:
        if not case.get("region_name") or not case.get("start_year") or not case.get("end_year"):
            continue
        case["case_id"] = slugify(case["region_name"], case["start_year"], case["end_year"])
        case_service.upsert_curated_case(case)
        logger.info(f"저장: {case['case_id']}")
        saved += 1

    logger.info(f"총 {len(cases)}건 발굴, {saved}건 저장 → {settings.chroma_db_dir}")


if __name__ == "__main__":
    main()
