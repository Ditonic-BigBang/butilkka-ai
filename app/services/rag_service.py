from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """FAISS 기반 RAG 서비스 (요청별 인메모리)"""

    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self._vectorstore: Optional[FAISS] = None

    def create_index(self, texts: list[str], metadatas: list[dict] = None) -> int:
        """텍스트로 FAISS 인덱스 생성"""
        if not texts:
            logger.warning("빈 텍스트 리스트로 인덱스 생성 시도")
            return 0

        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]

        self._vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        logger.info(f"FAISS 인덱스 생성: {len(texts)}건")
        return len(texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """유사 문서 검색"""
        if not self._vectorstore:
            logger.warning("인덱스가 없습니다. create_index를 먼저 호출하세요.")
            return []

        results = self._vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in results
        ]

    def get_context(self, query: str, k: int = 3) -> str:
        """검색 결과를 컨텍스트 문자열로 변환"""
        results = self.search(query, k=k)
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['content']}")

        return "\n\n".join(context_parts)


def create_rag_service(openai_api_key: str) -> RAGService:
    return RAGService(openai_api_key=openai_api_key)
