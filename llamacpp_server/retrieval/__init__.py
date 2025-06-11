"""Модуль для работы с RAG (Retrieval-Augmented Generation)."""

# Legacy imports (deprecated)
from .embedding_service import EmbeddingService
from .faiss_store import FaissVectorStore
from .semantic_chunker import SemanticChunker
from .rag_service import RAGService
from .rag_builder import RAGBuilder, build_rag_from_confluence
from .protocols import EmbeddingServiceProtocol, RAGServiceProtocol, VectorStoreProtocol

# Modern imports (recommended)
from .modern_chunker import ModernSemanticChunker, ChunkingStrategy
from .modern_embeddings import ModernEmbeddingService, EmbeddingProvider
from .modern_vector_store import ModernVectorStoreFactory, VectorStoreType, VectorStoreProtocol
from .modern_rag_service import ModernRAGService
from .modern_rag_builder import ModernRAGBuilder, build_modern_rag_from_confluence

__all__ = [
    # Legacy (deprecated - будут удалены в следующих версиях)
    "EmbeddingService",
    "FaissVectorStore", 
    "SemanticChunker",
    "RAGService",
    "RAGBuilder",
    "build_rag_from_confluence",
    
    # Protocols
    "EmbeddingServiceProtocol",
    "VectorStoreProtocol", 
    "RAGServiceProtocol",
    
    # Modern (recommended)
    "ModernSemanticChunker",
    "ChunkingStrategy",
    "ModernEmbeddingService",
    "EmbeddingProvider",
    "VectorStoreProtocol",
    "VectorStoreType",
    "ModernRAGService",
    "ModernRAGBuilder",
    "build_modern_rag_from_confluence",
]
