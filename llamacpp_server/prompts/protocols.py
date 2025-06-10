"""Протоколы для RAG Prompt сервисов."""

from typing import Protocol


class PromptServiceProtocol(Protocol):
    """Протокол для RAG сервиса работы с промптами."""

    async def get_system_prompt(self) -> str:
        """Получить системный промпт для RAG."""
        ...

    async def create_rag_prompt(self, query: str, context: list[str]) -> str:
        """Создать RAG промпт с контекстом и вопросом."""
        ...
