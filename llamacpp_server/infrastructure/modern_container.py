"""Современный DI контейнер для зависимостей с поддержкой modern RAG компонентов."""

from dependency_injector import containers, providers

from ..config import get_settings
from ..llama.factory import LlamaFactory
from ..llama.service import LlamaService
from ..prompts.service import PromptService

# Старые компоненты (для обратной совместимости)
from ..retrieval.embedding_service import EmbeddingService
from ..retrieval.faiss_store import FaissVectorStore
from ..retrieval.rag_service import RAGService

# Современные компоненты
from ..retrieval.modern_embeddings import ModernEmbeddingService, EmbeddingProvider
from ..retrieval.modern_vector_store import ModernVectorStoreFactory, VectorStoreType
from ..retrieval.modern_rag_service import ModernRAGService
from ..retrieval.modern_chunker import ModernSemanticChunker, ChunkingStrategy


class ModernContainer(containers.DeclarativeContainer):
    """Современный DI контейнер для всех зависимостей приложения с modern RAG компонентами."""

    # Конфигурация
    config = providers.Singleton(get_settings)

    # Factory для создания LLama
    llama_factory = providers.Factory(LlamaFactory)

    # LLama instance будет переопределен при старте
    llama_instance = providers.Singleton(None)

    # =================================
    # СОВРЕМЕННЫЕ RAG КОМПОНЕНТЫ
    # =================================
    
    # Современный семантический чанкер
    modern_chunker = providers.Singleton(
        ModernSemanticChunker,
        strategy=ChunkingStrategy.HYBRID,  # Гибридная стратегия по умолчанию
        chunk_size=512,
        overlap=50
    )

    # Современный эмбеддинг сервис
    modern_embedding_service = providers.Singleton(
        ModernEmbeddingService,
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model_name="BAAI/bge-m3"
    )

    # Современное векторное хранилище
    modern_vector_store = providers.Singleton(
        ModernVectorStoreFactory.create_store,
        store_type="chroma",
        collection_name="rag_documents"
    )

    # Современный RAG сервис
    modern_rag_service = providers.Singleton(
        ModernRAGService,
        embedding_service=modern_embedding_service,
        vector_store=modern_vector_store,
        settings=config
    )

    # =================================
    # СТАРЫЕ RAG КОМПОНЕНТЫ (для обратной совместимости)
    # =================================
    
    # Старые RAG компоненты - все как Singleton для инициализации один раз при старте
    legacy_embedding_service = providers.Singleton(
        EmbeddingService,
        settings=config
    )
    legacy_vector_store = providers.Singleton(
        FaissVectorStore,
        settings=config
    )
    legacy_rag_service = providers.Singleton(
        RAGService,
        embedding_service=legacy_embedding_service,
        vector_store=legacy_vector_store,
        settings=config
    )

    # =================================
    # ОСНОВНЫЕ СЕРВИСЫ
    # =================================

    # Prompt сервис
    prompt_service = providers.Singleton(
        PromptService,
        settings=config
    )

    # Сервисы - тоже Singleton для избежания повторных инициализаций
    # По умолчанию используем современный RAG сервис
    llama_service = providers.Singleton(
        LlamaService,
        llama=llama_instance,
        rag_service=modern_rag_service,  # Используем современный RAG сервис
        prompt_service=prompt_service,
        settings=config,
    )

    # Альтернативный сервис со старым RAG (для миграции)
    llama_service_legacy = providers.Singleton(
        LlamaService,
        llama=llama_instance,
        rag_service=legacy_rag_service,  # Используем старый RAG сервис
        prompt_service=prompt_service,
        settings=config,
    )


class HybridContainer(containers.DeclarativeContainer):
    """
    Гибридный контейнер для постепенной миграции.
    
    Позволяет переключаться между старыми и современными компонентами
    через настройки конфигурации.
    """

    # Конфигурация
    config = providers.Singleton(get_settings)

    # Factory для создания LLama
    llama_factory = providers.Factory(LlamaFactory)
    llama_instance = providers.Singleton(None)

    # Современные компоненты
    modern_embedding_service = providers.Singleton(
        ModernEmbeddingService,
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model_name="BAAI/bge-m3"
    )
    modern_vector_store = providers.Singleton(
        ModernVectorStoreFactory.create_store,
        store_type="chroma",
        collection_name="rag_documents"
    )
    modern_rag_service = providers.Singleton(
        ModernRAGService,
        embedding_service=modern_embedding_service,
        vector_store=modern_vector_store,
        settings=config
    )

    # Старые компоненты
    legacy_embedding_service = providers.Singleton(
        EmbeddingService,
        settings=config
    )
    legacy_vector_store = providers.Singleton(
        FaissVectorStore,
        settings=config
    )
    legacy_rag_service = providers.Singleton(
        RAGService,
        embedding_service=legacy_embedding_service,
        vector_store=legacy_vector_store,
        settings=config
    )

    # Prompt сервис
    prompt_service = providers.Singleton(
        PromptService,
        settings=config
    )

    # Динамический выбор RAG сервиса на основе настроек
    @providers.Singleton
    def rag_service(self):
        """Динамически выбираем RAG сервис на основе настроек."""
        settings = self.config()
        use_modern_rag = getattr(settings, 'use_modern_rag', True)
        
        if use_modern_rag:
            return self.modern_rag_service()
        else:
            return self.legacy_rag_service()

    # Основной LLama сервис с динамическим RAG
    llama_service = providers.Singleton(
        LlamaService,
        llama=llama_instance,
        rag_service=rag_service,
        prompt_service=prompt_service,
        settings=config,
    )


# Обратная совместимость - экспортируем старый Container
class Container(containers.DeclarativeContainer):
    """Оригинальный DI контейнер для обратной совместимости."""

    # Конфигурация
    config = providers.Singleton(get_settings)

    # Factory для создания LLama
    llama_factory = providers.Factory(LlamaFactory)
    llama_instance = providers.Singleton(None)

    # RAG компоненты - все как Singleton для инициализации один раз при старте
    embedding_service = providers.Singleton(
        EmbeddingService,
        settings=config
    )
    vector_store = providers.Singleton(
        FaissVectorStore,
        settings=config
    )
    rag_service = providers.Singleton(
        RAGService,
        embedding_service=embedding_service,
        vector_store=vector_store,
        settings=config
    )

    # Prompt сервис
    prompt_service = providers.Singleton(
        PromptService,
        settings=config
    )

    # Сервисы - тоже Singleton для избежания повторных инициализаций
    llama_service = providers.Singleton(
        LlamaService,
        llama=llama_instance,
        rag_service=rag_service,
        prompt_service=prompt_service,
        settings=config,
    ) 