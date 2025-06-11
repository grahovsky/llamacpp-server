"""Протоколы для RAG системы."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class EmbeddingProvider(Enum):
    """Провайдеры эмбеддингов."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    MODEL2VEC = "model2vec"
    OPENAI = "openai"


class VectorStoreType(Enum):
    """Типы векторных хранилищ."""
    CHROMADB = "chroma"


@dataclass
class Document:
    """Документ для поиска."""
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Результат поиска."""
    document: Document
    score: float


class EmbeddingServiceProtocol(Protocol):
    """Протокол для сервиса эмбеддингов."""

    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов."""
        ...


class VectorStoreProtocol(Protocol):
    """Протокол для векторного хранилища."""

    async def add_documents(self, documents: list[Document]) -> None:
        """Добавить документы в хранилище."""
        ...

    async def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        """Поиск похожих документов."""
        ...

    async def load_index(self, index_path: str) -> None:
        """Загрузить существующий индекс."""
        ...

    async def save_index(self, index_path: str) -> None:
        """Сохранить индекс."""
        ...


class RAGServiceProtocol(Protocol):
    """Протокол для RAG сервиса."""

    async def search_relevant_context(
        self, query: str, k: int = 5
    ) -> list[str]:
        """Найти релевантный контекст для запроса с citation-focused форматированием."""
        ...

    async def enhance_prompt_with_context(
        self, original_prompt: str, context: list[str]
    ) -> str:
        """Улучшить промпт контекстной информацией."""
        ...
