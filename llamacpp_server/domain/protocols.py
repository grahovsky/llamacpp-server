"""Протоколы для dependency inversion principle."""

from typing import AsyncIterator, Protocol

from .models import ChatCompletionRequest, CompletionResponse, TextCompletionRequest


class LlamaServiceProtocol(Protocol):
    """Протокол для работы с LLama моделью."""
    
    async def chat_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Генерация chat completion."""
        ...
    
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг chat completion."""
        ...
    
    async def text_completion(self, request: TextCompletionRequest) -> CompletionResponse:
        """Генерация text completion."""
        ...
    
    async def text_completion_stream(
        self, request: TextCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг text completion."""
        ...
    
    async def is_ready(self) -> bool:
        """Проверка готовности модели."""
        ...


class PromptServiceProtocol(Protocol):
    """Протокол для работы с промптами."""
    
    async def get_system_prompt(self, prompt_type: str) -> str:
        """Получить системный промпт."""
        ...
    
    async def format_chat_prompt(self, messages: list) -> str:
        """Форматировать промпт для чата."""
        ... 