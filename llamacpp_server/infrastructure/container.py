"""DI контейнер для зависимостей."""

from dependency_injector import containers, providers

from ..config import Settings, get_settings
from ..llama.factory import LlamaFactory
from ..llama.service import LlamaService
from ..prompts.service import PromptService


class Container(containers.DeclarativeContainer):
    """DI контейнер для всех зависимостей приложения."""
    
    # Конфигурация
    config = providers.Singleton(get_settings)
    
    # Factory для создания LLama
    llama_factory = providers.Factory(LlamaFactory)
    
    # LLama instance будет переопределен при старте
    llama_instance = providers.Singleton(None)
    
    # Сервисы
    llama_service = providers.Factory(
        LlamaService,
        llama=llama_instance,
    )
    
    prompt_service = providers.Factory(PromptService) 