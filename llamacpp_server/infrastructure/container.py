"""DI контейнер для зависимостей."""

from dependency_injector import containers, providers

from ..config import Settings, get_settings
from ..llama.factory import LlamaFactory
from ..llama.service import LlamaService
from ..prompts.service import PromptService
from ..retrieval.embedding_service import EmbeddingService
from ..retrieval.faiss_store import FaissVectorStore
from ..retrieval.rag_service import RAGService


class Container(containers.DeclarativeContainer):
    """DI контейнер для всех зависимостей приложения."""
    
    # Конфигурация
    config = providers.Singleton(get_settings)
    
    # Factory для создания LLama
    llama_factory = providers.Factory(LlamaFactory)
    
    # LLama instance будет переопределен при старте
    llama_instance = providers.Singleton(None)
    
    # RAG компоненты - все как Singleton для инициализации один раз при старте
    embedding_service = providers.Singleton(EmbeddingService)
    vector_store = providers.Singleton(FaissVectorStore)
    rag_service = providers.Singleton(
        RAGService,
        embedding_service=embedding_service,
        vector_store=vector_store
    )
    
    # Сервисы - тоже Singleton для избежания повторных инициализаций
    llama_service = providers.Singleton(
        LlamaService,
        llama=llama_instance,
        rag_service=rag_service,
    )
    
    prompt_service = providers.Singleton(PromptService) 