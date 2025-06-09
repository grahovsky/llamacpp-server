"""RAG сервис для контекстного поиска."""

from pathlib import Path

import structlog

from ..config.settings import get_settings
from ..prompts.templates import RAG_TEMPLATES
from .protocols import EmbeddingServiceProtocol, VectorStoreProtocol

logger = structlog.get_logger(__name__)


class RAGService:
    """Сервис для Retrieval-Augmented Generation."""

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        vector_store: VectorStoreProtocol
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        settings = get_settings()
        self._index_path = Path(settings.faiss_index_path)
        self._search_k = settings.rag_search_k
        self._max_context_length = settings.rag_max_context

        # Инициализируем компоненты
        self._embedding_model = None
        self._faiss_index = None
        self._documents = []

        logger.info("RAG сервис инициализирован",
                   max_context_length=self._max_context_length)

    async def enhance_prompt_with_context(
        self, original_prompt: str, context: list[str]
    ) -> str:
        """Улучшить промпт контекстной информацией."""
        if not context:
            logger.debug("Нет контекста для улучшения промпта")
            return original_prompt

        # Объединяем контекст с разделителями
        context_text = "\n\n".join(context)

        # Используем citation_focused_prompt как стандартное поведение
        enhanced_prompt = RAG_TEMPLATES["citation_focused_prompt"].format(
            context=context_text,
            question=original_prompt
        )

        logger.debug("Промпт улучшен контекстом",
                    original_len=len(original_prompt),
                    enhanced_len=len(enhanced_prompt),
                    context_docs=len(context),
                    template_used="citation_focused_prompt")

        # Детальное логирование для диагностики
        logger.debug("🔍 RAG Enhanced Prompt Preview",
                    enhanced_preview=enhanced_prompt[:800] + "..." if len(enhanced_prompt) > 800 else enhanced_prompt)

        return enhanced_prompt

    async def search_relevant_context(
        self, query: str, k: int = 5
    ) -> list[str]:
        """Найти релевантный контекст для запроса с citation-focused форматированием."""
        logger.info("🔍 RAG поиск контекста с цитированием", query_preview=query[:100], k=k)

        try:
            # Получаем эмбеддинг запроса
            query_embedding = await self._embedding_service.embed_text(query)
            logger.debug("Эмбеддинг создан", embedding_dim=len(query_embedding))

            # Ищем релевантные документы
            search_results = await self._vector_store.search(query_embedding, k=k)
            logger.info("Результаты поиска получены", found_docs=len(search_results))

            if not search_results:
                logger.warning("❌ Не найдено релевантных документов")
                return []

            # Форматируем контекст с особым акцентом на источники
            context_docs = []
            total_length = 0
            unique_sources = set()

            for _, result in enumerate(search_results):
                document = result.document
                content = document.content
                metadata = document.metadata

                title = metadata.get('title', 'Без названия')
                source_url = metadata.get('source', '')

                # Усиленный формат для цитирования
                if source_url:
                    unique_sources.add((title, source_url))
                    formatted_doc = f"""Документ: {title}
URL: {source_url}
Содержание: {content}

"""
                else:
                    formatted_doc = f"""Документ: {title}
URL: [источник неизвестен]
Содержание: {content}

"""

                # Проверяем лимит длины
                if total_length + len(formatted_doc) > self._max_context_length:
                    remaining = self._max_context_length - total_length
                    if remaining > 150:  # Минимальный размер для полезного контента
                        # Сохраняем важные части
                        header = f"Документ: {title}\nURL: {source_url}\n"
                        footer = "\n\n"
                        content_space = remaining - len(header) - len(footer) - len('Содержание: ') - 20

                        if content_space > 80:
                            truncated_content = content[:content_space] + "...[ОБРЕЗАНО]"
                            formatted_doc = f'{header}Содержание: {truncated_content}{footer}'

                            context_docs.append(formatted_doc)
                            total_length += len(formatted_doc)
                            if source_url:
                                unique_sources.add((title, source_url))
                    break

                context_docs.append(formatted_doc)
                total_length += len(formatted_doc)

            sources_list = [f"{title} - {url}" for title, url in unique_sources]

            logger.info("✅ RAG контекст с цитированием готов",
                       documents=len(context_docs),
                       total_chars=total_length,
                       unique_sources=len(sources_list),
                       sources=sources_list)

            return context_docs

        except Exception as e:
            logger.error("❌ Ошибка RAG поиска с цитированием", error=str(e), exc_info=True)
            return []
