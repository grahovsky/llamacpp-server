"""RAG сервис для контекстного поиска."""

from pathlib import Path
from typing import Any

import structlog

from ..config.settings import get_settings
from ..prompts.templates import get_prompt_template
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

        # Получаем шаблон промпта на основе типа модели
        self._model_type = settings.get_model_type()
        self._prompt_template = get_prompt_template(self._model_type)

        # Инициализируем компоненты
        self._embedding_model = None
        self._faiss_index = None
        self._documents = []

        logger.info("RAG сервис инициализирован",
                   max_context_length=self._max_context_length,
                   model_type=self._model_type)

    async def create_rag_prompt(self, user_query: str) -> str:
        """Создать RAG промпт для любого пользовательского запроса (новый метод для RAG-only)."""
        logger.info("🧠 Создание RAG промпта", query_preview=user_query[:100])

        try:
            # Ищем релевантный контекст
            context_docs = await self.search_relevant_context(user_query, k=self._search_k)
            
            if not context_docs:
                logger.warning("⚠️ Контекст не найден, используем пустой контекст")
                context_text = "Релевантная документация не найдена."
            else:
                context_text = "\n\n".join(context_docs)
                logger.info("✅ Контекст найден", docs_count=len(context_docs))

            # Создаем RAG промпт используя новый шаблон
            rag_prompt = self._prompt_template.format_rag_prompt(
                query=user_query,
                context=context_docs
            )

            logger.debug("RAG промпт создан",
                        original_len=len(user_query),
                        rag_len=len(rag_prompt),
                        context_docs=len(context_docs))

            return rag_prompt

        except Exception as e:
            logger.error("❌ Ошибка создания RAG промпта", error=str(e), exc_info=True)
            # Fallback: возвращаем промпт с пустым контекстом
            return self._prompt_template.format_rag_prompt(
                query=user_query,
                context=["Ошибка поиска в документации."]
            )

    async def enhance_prompt_with_context(
        self, original_prompt: str, context: list[str]
    ) -> str:
        """Улучшить промпт контекстной информацией."""
        if not context:
            logger.debug("Нет контекста для улучшения промпта")
            return original_prompt

        # Объединяем контекст с разделителями
        context_text = "\n\n".join(context)

        # Используем шаблон на основе типа модели
        enhanced_prompt = self._prompt_template.format_rag_prompt(
            query=original_prompt,
            context=context
        )

        logger.debug("Промпт улучшен контекстом",
                    original_len=len(original_prompt),
                    enhanced_len=len(enhanced_prompt),
                    context_docs=len(context),
                    model_type=self._model_type)

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

            # Простое форматирование - используем контент как есть
            context_docs = []
            total_length = 0
            unique_sources = set()

            for _, result in enumerate(search_results):
                document = result.document
                content = document.content  # Используем контент как есть (уже с контекстом)
                metadata = document.metadata
                
                title = metadata.get('title', 'Без названия')
                source_url = metadata.get('source', '')
                
                # Собираем уникальные источники для логирования
                if source_url:
                    unique_sources.add((title, source_url))

                # Проверяем лимит длины
                if total_length + len(content) > self._max_context_length:
                    remaining = self._max_context_length - total_length
                    if remaining > 100:  # Минимальный размер для полезного контента
                        truncated_content = content[:remaining-20] + "...[ОБРЕЗАНО]"
                        context_docs.append(truncated_content)
                        total_length += len(truncated_content)
                        if source_url:
                            unique_sources.add((title, source_url))
                    break

                context_docs.append(content)
                total_length += len(content)

            sources_list = [f"{title} - {url}" for title, url in unique_sources]

            logger.info("✅ RAG контекст готов",
                       documents=len(context_docs),
                       total_chars=total_length,
                       unique_sources=len(sources_list),
                       sources=sources_list)

            return context_docs

        except Exception as e:
            logger.error("❌ Ошибка RAG поиска с цитированием", error=str(e), exc_info=True)
            return []

    async def batch_add_documents(
        self, 
        documents: list, 
        batch_size: int = 50,
        show_progress: bool = True
    ) -> dict[str, Any]:
        """Добавить документы батчами с прогресс-индикатором."""
        logger.info("📦 Начинаем массовое добавление документов",
                   total_docs=len(documents),
                   batch_size=batch_size)

        if not documents:
            return {"success": False, "error": "Нет документов для добавления"}

        try:
            # Статистика процесса
            stats = {
                "total_docs": len(documents),
                "processed_docs": 0,
                "failed_docs": 0,
                "batches_processed": 0,
                "total_batches": (len(documents) + batch_size - 1) // batch_size,
                "embeddings_created": 0,
                "errors": []
            }

            # Обрабатываем документы батчами
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_num = stats["batches_processed"] + 1
                
                if show_progress:
                    logger.info(f"🔄 Обработка батча {batch_num}/{stats['total_batches']}",
                               docs_in_batch=len(batch_docs))

                try:
                    # Создаем эмбеддинги для батча
                    batch_embeddings = []
                    for doc in batch_docs:
                        try:
                            embedding = await self._embedding_service.embed_text(doc.content)
                            doc.embedding = embedding
                            batch_embeddings.append(embedding)
                            stats["embeddings_created"] += 1
                        except Exception as e:
                            logger.error("Ошибка создания эмбеддинга", 
                                       doc_id=getattr(doc, 'id', 'unknown'),
                                       error=str(e))
                            stats["failed_docs"] += 1
                            stats["errors"].append(f"Embedding error for doc {getattr(doc, 'id', 'unknown')}: {str(e)}")
                            continue

                    # Добавляем документы с эмбеддингами в векторное хранилище
                    valid_docs = [doc for doc in batch_docs if hasattr(doc, 'embedding') and doc.embedding is not None]
                    
                    if valid_docs:
                        await self._vector_store.add_documents(valid_docs, batch_size=len(valid_docs))
                        stats["processed_docs"] += len(valid_docs)
                    
                    stats["batches_processed"] += 1

                    if show_progress:
                        progress_pct = (stats["processed_docs"] / stats["total_docs"]) * 100
                        logger.info(f"✅ Батч {batch_num} завершен",
                                   progress=f"{progress_pct:.1f}%",
                                   processed=stats["processed_docs"],
                                   total=stats["total_docs"])

                except Exception as e:
                    logger.error(f"Ошибка обработки батча {batch_num}", error=str(e))
                    stats["errors"].append(f"Batch {batch_num} error: {str(e)}")
                    stats["failed_docs"] += len(batch_docs)

            # Финальная статистика
            success_rate = (stats["processed_docs"] / stats["total_docs"]) * 100 if stats["total_docs"] > 0 else 0
            
            logger.info("🎉 Массовое добавление завершено",
                       total_docs=stats["total_docs"],
                       processed=stats["processed_docs"],
                       failed=stats["failed_docs"],
                       success_rate=f"{success_rate:.1f}%",
                       embeddings_created=stats["embeddings_created"])

            return {
                "success": True,
                "stats": stats,
                "success_rate": success_rate
            }

        except Exception as e:
            logger.error("❌ Критическая ошибка массового добавления", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "stats": stats if 'stats' in locals() else {}
            }
