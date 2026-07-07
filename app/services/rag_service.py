from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 벡터 임베딩 서비스"""

    def __init__(
        self,
        openai_api_key: str,
        persist_directory: str = "chroma_db"
    ):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """벡터스토어 (lazy loading)"""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self._vectorstore

    def add_documents(self, texts: list[str], metadatas: list[dict] = None) -> int:
        """문서 추가 및 임베딩"""
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]

        # 청크 분할
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Adding {len(chunks)} chunks to vectorstore")

        # 벡터스토어에 추가
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict = None
    ) -> list[dict]:
        """유사 문서 검색"""
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in results
        ]

    def generate_context(self, query: str, k: int = 3) -> str:
        """검색 결과를 컨텍스트 문자열로 변환"""
        results = self.search(query, k=k)
        context_parts = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['content']}")

        return "\n\n".join(context_parts)


# Factory function (API key 필요하므로 singleton 대신)
def create_rag_service(openai_api_key: str) -> RAGService:
    return RAGService(openai_api_key=openai_api_key)
