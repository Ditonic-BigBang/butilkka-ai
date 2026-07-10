"""큐레이션 사례 시드 데이터를 Chroma 벡터스토어에 색인.

수동 실행: python -m app.scripts.build_case_index
재실행해도 delete-then-add(idempotent)라 중복 색인되지 않는다.
"""
import json
import logging

from app.core.config import get_settings
from app.services.case_service import get_case_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED_PATH = "app/data/case_studies.json"


def main():
    settings = get_settings()
    case_service = get_case_service(
        openai_api_key=settings.openai_api_key,
        persist_directory=settings.chroma_db_dir,
    )

    with open(SEED_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        case_service.upsert_curated_case(case)
        logger.info(f"색인 완료: {case['case_id']}")

    logger.info(f"총 {len(cases)}건 색인 완료 → {settings.chroma_db_dir}")


if __name__ == "__main__":
    main()
