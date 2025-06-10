"""Сервис для работы с промптами."""


import structlog

from ..config.settings import get_settings
from ..domain.models import ChatMessage
from .templates import get_prompt_template

logger = structlog.get_logger(__name__)


class PromptService:
    """Сервис для работы с промптами."""

    def __init__(self) -> None:
        """Инициализация сервиса промптов."""
        settings = get_settings()
        self._model_type = settings.get_model_type()
        self._prompt_template = get_prompt_template(self._model_type)
        
        logger.info("Промпт сервис инициализирован", model_type=self._model_type)

    async def get_system_prompt(self, prompt_type: str = "default") -> str:
        """Получить системный промпт."""
        logger.debug("Получение системного промпта", prompt_type=prompt_type)
        
        # Базовый системный промпт для RAG
        default_system_prompt = (
            "Ты — ассистент, который помогает отвечать на вопросы на основе предоставленной документации. "
            "Используй только информацию из контекста для ответа. "
            "Если информации недостаточно, так и скажи."
        )
        
        return default_system_prompt

    async def format_chat_prompt(
        self, messages: list[ChatMessage], system_message: str = ""
    ) -> str:
        """Форматировать промпт для чата."""
        logger.debug("Форматирование промпта", 
                    model_type=self._model_type, 
                    messages_count=len(messages))

        # Конвертируем ChatMessage в словари
        message_dicts = []
        for msg in messages:
            message_dicts.append({
                "role": msg.role,
                "content": msg.content
            })

        # Используем шаблон на основе типа модели
        return self._prompt_template.format_chat_prompt(
            messages=message_dicts,
            system_message=system_message
        )

    async def create_conversation_prompt(
        self, messages: list[ChatMessage], system_message: str = ""
    ) -> str:
        """Создать промпт для многооборотной беседы."""
        logger.debug("Создание промпта для беседы", model_type=self._model_type)
        
        # Используем тот же метод что и для чата
        return await self.format_chat_prompt(messages, system_message)

    async def format_rag_prompt(self, query: str, context: list[str], system_message: str = "") -> str:
        """Форматировать RAG промпт."""
        logger.debug("Форматирование RAG промпта", 
                    model_type=self._model_type,
                    query_len=len(query),
                    context_docs=len(context))
        
        return self._prompt_template.format_rag_prompt(
            query=query,
            context=context,
            system_message=system_message
        )
