"""Сервис для работы с промптами."""

import structlog
from typing import List

from .templates import CHAT_TEMPLATES, SYSTEM_PROMPTS
from ..domain.models import ChatMessage


logger = structlog.get_logger(__name__)


class PromptService:
    """Сервис для работы с промптами."""
    
    async def get_system_prompt(self, prompt_type: str) -> str:
        """Получить системный промпт."""
        logger.debug("Получение системного промпта", prompt_type=prompt_type)
        return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])
    
    async def format_chat_prompt(
        self, messages: List[ChatMessage], template: str = "llama2"
    ) -> str:
        """Форматировать промпт для чата."""
        logger.debug("Форматирование промпта", template=template, messages_count=len(messages))
        
        # Найти системный промпт
        system_prompt = ""
        user_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_messages.append(msg.content)
        
        # Использовать дефолтный системный промпт если не задан
        if not system_prompt:
            system_prompt = SYSTEM_PROMPTS["default"]
        
        # Объединить пользовательские сообщения
        user_message = "\n".join(user_messages)
        
        # Применить шаблон
        template_str = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES["llama2"])
        
        return template_str.format(
            system_prompt=system_prompt,
            user_message=user_message,
        )
    
    async def create_conversation_prompt(
        self, messages: List[ChatMessage], template: str = "llama2"
    ) -> str:
        """Создать промпт для многооборотной беседы."""
        logger.debug("Создание промпта для беседы", template=template)
        
        # Для многооборотной беседы используем более сложное форматирование
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                formatted_messages.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_messages.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_messages.append(f"Assistant: {msg.content}")
        
        # Добавляем префикс для ответа ассистента
        formatted_messages.append("Assistant:")
        
        return "\n".join(formatted_messages) 