"""Сервис для работы с промптами в RAG-only системе."""

import structlog

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class PromptService:
    """Сервис для работы с промптами в RAG-only системе."""

    def __init__(self, settings=None) -> None:
        """Инициализация сервиса промптов."""
        if settings is None:
            settings = get_settings()

        self._settings = settings
        logger.info("RAG Промпт сервис инициализирован")

    async def get_system_prompt(self) -> str:
        """Получить системный промпт для RAG."""
        logger.debug("Получение системного RAG промпта")

        return (
            "Ты помощник для ответов на вопросы на основе предоставленной документации.\n\n"
            "Правила:\n"
            "- Отвечай на русском языке подробно и развернуто на основе предоставленной документации\n"
            "- Если в документах абсолютно нет релевантной информации, скажи \"Информация не найдена в документации\" и закончи ответ\n"
            "- Всегда включай самые релевантные ссылки из документации в начале ответа в разделе \"Источники:\"\n"
            "- Используй всю доступную информацию из документов для формирования полного ответа\n"
            "- НЕ выдумывай факты, но интерпретируй и объясняй найденную информацию\n"
        )
            # "- Отвечай только на заданный вопрос, не добавляй лишнего"

    async def create_rag_prompt(self, query: str, context: list[str]) -> str:
        """Создать структурированный RAG промпт для Llama 3.1."""
        logger.debug("Создание RAG промпта",
                    query_len=len(query),
                    context_docs=len(context))

        # Получаем системный промпт
        system_prompt = await self.get_system_prompt()

        context_text = "\n\n".join(context) if context else "Контекст не найден."

        # Используем структуру Llama 3.1 с четким разделением ролей
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Документация:
{context_text}

Вопрос: {query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        return prompt
