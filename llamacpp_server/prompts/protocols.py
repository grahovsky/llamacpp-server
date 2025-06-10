"""Протоколы для Prompt сервисов."""

from typing import Protocol


class PromptServiceProtocol(Protocol):
    """Протокол для сервиса работы с промптами."""

    def get_system_prompt(self) -> str:
        """Получить системный промпт."""
        ...

    def create_rag_prompt(self, query: str, context: str) -> str:
        """Создать RAG промпт."""
        ... 