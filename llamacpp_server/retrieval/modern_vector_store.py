"""Современное векторное хранилище с поддержкой ChromaDB."""

import asyncio
import uuid
from abc import abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import structlog

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from .protocols import Document, SearchResult

logger = structlog.get_logger(__name__)


class VectorStoreType(Enum):
    """Типы векторных хранилищ."""
    CHROMADB = "chroma"


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Протокол для векторных хранилищ."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> None:
        """Добавить документы в хранилище."""
        ...

    @abstractmethod
    async def search(self, query_embedding: list[float], k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        """Поиск похожих документов."""
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        """Удалить коллекцию."""
        ...

    @abstractmethod
    async def get_collection_info(self) -> dict[str, Any]:
        """Получить информацию о коллекции."""
        ...


class ChromaVectorStore:
    """Векторное хранилище на основе ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_path: str | None = None,
        **kwargs
    ):
        if not HAS_CHROMA:
            raise ImportError("chromadb package не установлен. Установите: pip install chromadb")

        self.collection_name = collection_name
        self.persist_path = persist_path

        # Создаем клиент в зависимости от наличия persist_path
        if persist_path:
            # Persistent клиент для сохранения на диск
            self.client = chromadb.PersistentClient(path=persist_path)
            logger.info("ChromaDB клиент создан с persist_path", path=persist_path)
        else:
            # Ephemeral клиент (в памяти)
            self.client = chromadb.EphemeralClient()
            logger.info("ChromaDB клиент создан в памяти (ephemeral)")

        self.collection = None

    async def _ensure_collection(self) -> None:
        """Инициализация коллекции."""
        if self.collection is None:
            logger.info("Инициализация ChromaDB коллекции", name=self.collection_name)

            loop = asyncio.get_event_loop()
            self.collection = await loop.run_in_executor(
                None,
                lambda: self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            )

    async def add_documents(self, documents: list[Document]) -> None:
        """Добавить документы в ChromaDB."""
        await self._ensure_collection()

        logger.info("Добавление документов в ChromaDB", count=len(documents))

        # Подготавливаем данные для ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []

        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"У документа {doc.id} отсутствует эмбеддинг")

            ids.append(doc.id)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata)
            documents_text.append(doc.content)

        # Добавляем в коллекцию
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
        )

        logger.info("Документы добавлены в ChromaDB")

    async def add_document(self, document: Document, embedding: list[float]) -> None:
        """Добавить один документ в ChromaDB."""
        # Устанавливаем эмбеддинг в документ
        document_with_embedding = Document(
            id=document.id,
            content=document.content,
            metadata=document.metadata,
            embedding=embedding
        )
        # Используем существующий метод для добавления
        await self.add_documents([document_with_embedding])

    async def search(self, query_embedding: list[float], k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        """Поиск в ChromaDB."""
        await self._ensure_collection()

        logger.debug("Поиск в ChromaDB", k=k, min_score=min_score)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"]
            )
        )

        # Конвертируем результаты
        search_results = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                embedding=None  # Не загружаем эмбеддинги для экономии памяти
            )

            # ChromaDB возвращает distance, конвертируем в similarity score
            distance = results["distances"][0][i]
            score = 1.0 - distance  # Для cosine distance

            # Фильтруем по минимальному score
            if score >= min_score:
                search_results.append(SearchResult(document=doc, score=score))

        logger.debug("Результаты после фильтрации", 
                    original_count=len(results["ids"][0]),
                    filtered_count=len(search_results))

        return search_results

    async def delete_collection(self) -> None:
        """Удалить коллекцию ChromaDB."""
        logger.info("Удаление ChromaDB коллекции", name=self.collection_name)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.delete_collection(self.collection_name)
        )

        self.collection = None

    async def get_collection_info(self) -> dict[str, Any]:
        """Получить информацию о коллекции ChromaDB."""
        await self._ensure_collection()

        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(
            None,
            lambda: self.collection.count()
        )

        return {
            "name": self.collection_name,
            "count": count,
            "type": "ChromaDB"
        }


class ModernVectorStoreFactory:
    """Фабрика для создания современных векторных хранилищ (только ChromaDB)."""

    @staticmethod
    def create_store(
        store_type: str,
        collection_name: str = "documents",
        **kwargs
    ) -> VectorStoreProtocol:
        """Создать векторное хранилище."""
        if store_type == "chroma":
            return ChromaVectorStore(collection_name=collection_name, **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый тип хранилища: {store_type}. Поддерживается только: chroma")

    @staticmethod
    def get_available_stores() -> list[str]:
        """Получить список доступных типов хранилищ."""
        available = []
        if HAS_CHROMA:
            available.append("chroma")
        return available


def create_vector_store(
    store_type: str = "chroma",
    collection_name: str = "documents",
    **kwargs
) -> VectorStoreProtocol:
    """
    Convenience функция для создания векторного хранилища.
    
    Args:
        store_type: Тип хранилища ("chroma")
        collection_name: Имя коллекции
        **kwargs: Дополнительные параметры
        
    Returns:
        Экземпляр векторного хранилища
    """
    return ModernVectorStoreFactory.create_store(
        store_type=store_type,
        collection_name=collection_name,
        **kwargs
    )
