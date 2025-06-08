"""Протоколы для RAG системы."""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Документ для поиска."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Результат поиска."""
    document: Document
    score: float
    

class EmbeddingServiceProtocol(Protocol):
    """Протокол для сервиса эмбеддингов."""
    
    async def embed_text(self, text: str) -> List[float]:
        """Получить эмбеддинг для текста."""
        ...
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для пакета текстов."""
        ...


class VectorStoreProtocol(Protocol):
    """Протокол для векторного хранилища."""
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Добавить документы в хранилище."""
        ...
    
    async def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
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
    ) -> List[str]:
        """Найти релевантный контекст для запроса."""
        ...
    
    async def enhance_prompt_with_context(
        self, original_prompt: str, context: List[str]
    ) -> str:
        """Улучшить промпт контекстной информацией."""
        ... 