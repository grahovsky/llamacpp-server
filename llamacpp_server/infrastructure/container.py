"""DI контейнер для зависимостей."""

from dependency_injector import containers, providers

from ..config import get_settings
from ..llama.factory import LlamaFactory
from ..llama.service import LlamaService
from ..prompts.service import PromptService
from ..retrieval.embedding_service import EmbeddingService
from ..retrieval.faiss_store import FaissVectorStore
from ..retrieval.rag_service import RAGService

# Современные RAG компоненты
from ..retrieval.modern_embeddings import ModernEmbeddingService
from ..retrieval.modern_rag_service import ModernRAGService
from ..retrieval.modern_vector_store import ModernVectorStoreFactory
from ..retrieval.protocols import EmbeddingProvider, VectorStoreType


def create_modern_embedding_service(settings):
    """Factory функция для создания современного embedding сервиса."""
    provider_map = {
        "sentence_transformers": EmbeddingProvider.SENTENCE_TRANSFORMERS,
        "model2vec": EmbeddingProvider.MODEL2VEC,
        "openai": EmbeddingProvider.OPENAI
    }
    
    provider = provider_map.get(
        settings.modern_embedding_provider, 
        EmbeddingProvider.SENTENCE_TRANSFORMERS
    )
    
    return ModernEmbeddingService(
        provider=provider,
        model_name=settings.modern_embedding_model,
        api_key=settings.openai_api_key if provider == EmbeddingProvider.OPENAI else None
    )


def create_modern_vector_store(settings):
    """Factory функция для создания современного векторного хранилища (только ChromaDB)."""
    config = {
        "collection_name": settings.modern_collection_name,
        "persist_path": settings.chromadb_path
    }
    
    return ModernVectorStoreFactory.create_store("chroma", **config)


def create_rag_service(settings, embedding_service, vector_store, prompt_service):
    """Factory функция для создания RAG сервиса (старого или современного)."""
    if settings.use_modern_rag:
        # Создаем новый современный RAG сервис
        provider_map = {
            "sentence_transformers": EmbeddingProvider.SENTENCE_TRANSFORMERS,
            "model2vec": EmbeddingProvider.MODEL2VEC,
            "openai": EmbeddingProvider.OPENAI
        }
        
        return ModernRAGService(
            embedding_service=None,  # Будет создан автоматически
            vector_store=None,  # Будет создан автоматически
            settings=settings,
            embedding_provider=provider_map.get(
                settings.modern_embedding_provider,
                EmbeddingProvider.SENTENCE_TRANSFORMERS
            ),
            embedding_model=settings.modern_embedding_model,
            vector_store_type=VectorStoreType.CHROMADB,  # Всегда ChromaDB
            collection_name=settings.modern_collection_name
        )
    else:
        # Используем старый RAG сервис
        return RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            settings=settings
        )


class Container(containers.DeclarativeContainer):
    """DI контейнер для всех зависимостей приложения."""

    # Конфигурация
    config = providers.Singleton(get_settings)

    # Factory для создания LLama
    llama_factory = providers.Factory(LlamaFactory)

    # LLama instance будет переопределен при старте
    llama_instance = providers.Singleton(None)

    # Prompt сервис (общий для старого и нового RAG)
    prompt_service = providers.Singleton(
        PromptService,
        settings=config
    )

    # Старые RAG компоненты (будут использоваться если use_modern_rag=False)
    embedding_service = providers.Singleton(
        EmbeddingService,
        settings=config
    )
    vector_store = providers.Singleton(
        FaissVectorStore,
        settings=config
    )

    # Современные RAG компоненты (будут использоваться если use_modern_rag=True)
    modern_embedding_service = providers.Factory(
        create_modern_embedding_service,
        settings=config
    )
    modern_vector_store = providers.Factory(
        create_modern_vector_store,
        settings=config
    )

    # RAG сервис с автоматическим переключением
    rag_service = providers.Factory(
        create_rag_service,
        settings=config,
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_service=prompt_service
    )

    # Сервисы - тоже Singleton для избежания повторных инициализаций
    llama_service = providers.Singleton(
        LlamaService,
        llama=llama_instance,
        rag_service=rag_service,
        prompt_service=prompt_service,
        settings=config,
    )
