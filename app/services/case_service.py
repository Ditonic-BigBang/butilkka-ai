from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CaseService:
    """Chroma 기반 유사사례 벡터 스토어 (영속, 실제 뉴스 기반으로 발굴한 확정된 과거 사례만 저장)"""

    COLLECTION_NAME = "commercial_district_cases"
    # 큐레이션 외부 사례가 공유하는 region_code (Spring regions 테이블에 1건만 미리 등록해두면
    # case_studies.json에 사례를 몇 개를 추가하든 추가 마이그레이션이 필요 없음)
    EXTERNAL_CASE_REGION_CODE = "EXT-CASE"

    def __init__(self, openai_api_key: str, persist_directory: str):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.persist_directory = persist_directory
        self._store: Optional[Chroma] = None

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._store

    def _upsert(self, doc_id: str, text: str, metadata: dict) -> None:
        """id 기준 delete 후 add (chromadb 버전별 upsert 차이를 피하기 위한 idempotent 구현)"""
        store = self._get_store()
        try:
            store.delete(ids=[doc_id])
        except Exception:
            pass  # 기존 id가 없으면 무시
        metadata = {**metadata, "doc_id": doc_id}  # 검색 결과 dedup용 (region_code가 공유될 수 있음)
        # Chroma는 메타데이터 값으로 None을 허용하지 않으므로 빈 문자열로 치환
        metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
        store.add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])

    def upsert_curated_case(self, case: dict) -> None:
        """case_studies.json 항목 하나를 임베딩·저장"""
        doc_id = f"case:{case['case_id']}"
        tags = case.get("tags") or []
        text = f"{case['region_name']} {case.get('district_name', '')} {case['decline_type']} " \
               f"{case['summary']} {case.get('description', '')} {' '.join(tags)}"

        padded_tags = (tags + [None, None, None, None])[:4]

        metadata = {
            "source": "curated_case",
            "region_code": case.get("region_code") or self.EXTERNAL_CASE_REGION_CODE,
            "region_name": case["region_name"],
            "district_name": case.get("district_name"),
            "decline_type": case["decline_type"],
            "start_year": case.get("start_year"),
            "end_year": case.get("end_year"),
            "summary": case["summary"],
            "description": case.get("description"),
            "tag1": padded_tags[0],
            "tag2": padded_tags[1],
            "tag3": padded_tags[2],
            "tag4": padded_tags[3],
        }
        self._upsert(doc_id, text, metadata)

    def search_similar(self, query: str, k: int, exclude_region_code: str) -> list[dict]:
        """시맨틱 유사도 검색 (큐레이션 사례만 대상 — 확정된 결과가 있는 사례만 유사사례로 씀.
        같은 region_code 제외, 중복 제거)"""
        store = self._get_store()
        try:
            raw = store.similarity_search_with_score(query, k=k * 3)
        except Exception as e:
            logger.warning(f"유사 사례 벡터 검색 실패: {e}")
            return []

        seen_ids = set()
        seen_display_regions = set()
        results = []
        for doc, score in sorted(raw, key=lambda x: x[1]):
            meta = doc.metadata
            if meta.get("source") != "curated_case":
                continue  # 과거 생성 리포트는 예측 스냅샷이라 확정된 사례가 아님 — 제외

            region_code = meta.get("region_code")
            if region_code == exclude_region_code:
                continue

            doc_id = meta.get("doc_id")
            if doc_id in seen_ids:
                continue

            # 응답에서는 구 단위로 표시되므로(_find_similar_cases가 district_name을 우선
            # 사용) 서로 다른 사례라도 같은 구 소속이면 같은 "지역"으로 보인다. 점수가
            # 더 좋은(= 먼저 등장하는, raw가 score 오름차순 정렬됨) 사례 하나만 남긴다.
            display_region = (meta.get("district_name") or meta.get("region_name") or "").strip()
            if display_region and display_region in seen_display_regions:
                continue

            seen_ids.add(doc_id)
            if display_region:
                seen_display_regions.add(display_region)

            # 저장 시 None -> "" 로 치환했던 것을 복원
            results.append({k: (None if v == "" else v) for k, v in meta.items()})
            if len(results) >= k:
                break

        return results


# Lazy singleton (Chroma는 영속 스토어이므로 요청마다 새로 열지 않음)
_case_service: Optional[CaseService] = None


def get_case_service(openai_api_key: str = None, persist_directory: str = None) -> CaseService:
    global _case_service
    if _case_service is None:
        from app.core.config import get_settings
        settings = get_settings()
        _case_service = CaseService(
            openai_api_key=openai_api_key or settings.openai_api_key,
            persist_directory=persist_directory or settings.chroma_db_dir,
        )
    return _case_service
