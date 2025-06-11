"""Современный RAG сервис для контекстного поиска."""

from typing import Any

import structlog

from ..config.settings import get_settings
from ..prompts.service import PromptService
from .modern_embeddings import EmbeddingProvider, ModernEmbeddingService
from .modern_vector_store import VectorStoreProtocol, VectorStoreType
from .protocols import Document

logger = structlog.get_logger(__name__)


class ModernRAGService:
    """Современный сервис для Retrieval-Augmented Generation с лучшими практиками 2025."""

    def __init__(
        self,
        embedding_service: ModernEmbeddingService | None = None,
        vector_store: VectorStoreProtocol | None = None,
        settings: Any | None = None,
        # Настройки для автоматической инициализации
        embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
        embedding_model: str = "BAAI/bge-m3",
        vector_store_type: VectorStoreType = VectorStoreType.CHROMADB,
        collection_name: str = "rag_documents"
    ) -> None:
        """
        Инициализация современного RAG сервиса.

        Args:
            embedding_service: Готовый сервис эмбеддингов (опционально)
            vector_store: Готовое векторное хранилище (опционально)
            settings: Настройки приложения
            embedding_provider: Провайдер эмбеддингов для автоинициализации
            embedding_model: Модель эмбеддингов для автоинициализации
            vector_store_type: Тип векторного хранилища для автоинициализации
            collection_name: Имя коллекции для автоинициализации
        """
        # Получаем настройки
        if settings is None:
            settings = get_settings()
        self._settings = settings

        # Сохраняем параметры для автоинициализации
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._vector_store_type = vector_store_type
        self._collection_name = collection_name

        # RAG настройки
        self._search_k = getattr(settings, 'rag_search_k', 5)
        self._max_context_length = getattr(settings, 'rag_max_context', 8000)
        self._min_score = getattr(settings, 'modern_min_score', 0.0)

        # Компоненты
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._prompt_service = PromptService(settings)

        # Флаг инициализации
        self._initialized = False

        logger.info("Современный RAG сервис создан",
                   embedding_provider=embedding_provider.value,
                   vector_store_type=vector_store_type.value,
                   collection_name=collection_name,
                   max_context_length=self._max_context_length,
                   min_score=self._min_score)

    async def initialize(self) -> None:
        """Инициализация компонентов при необходимости."""
        if self._initialized:
            return

        logger.info("🚀 Инициализация современного RAG сервиса")

        # Инициализируем embedding service если не передан
        if self._embedding_service is None:
            embedding_config = {
                "provider": self._embedding_provider,
                "model_name": self._embedding_model
            }

            if self._embedding_provider == EmbeddingProvider.OPENAI:
                embedding_config["api_key"] = self._settings.openai_api_key

            self._embedding_service = ModernEmbeddingService(**embedding_config)
            await self._embedding_service.initialize()
            logger.info("✅ Современный эмбеддинг сервис инициализирован")

        # Инициализируем vector store если не передан
        if self._vector_store is None:
            store_config = {
                "collection_name": self._collection_name,
                "persist_path": getattr(self._settings, 'chromadb_path', './data/chromadb')
            }

            from .modern_vector_store import ModernVectorStoreFactory
            
            self._vector_store = ModernVectorStoreFactory.create_store("chroma", **store_config)
            logger.info("✅ Современное векторное хранилище инициализировано")

        self._initialized = True
        logger.info("✅ Современный RAG сервис готов к работе")

    async def create_rag_prompt(self, user_query: str) -> str:
        """Создать современный RAG промпт для пользовательского запроса."""
        await self.initialize()

        logger.info("🧠 Создание современного RAG промпта", query_preview=user_query[:100])

        try:
            # Ищем релевантный контекст
            context_docs = await self.search_relevant_context(
                user_query, 
                k=self._search_k, 
                min_score=self._min_score
            )

            if not context_docs:
                logger.warning("⚠️ Контекст не найден, используем пустой контекст")
                context_list = ["Релевантная документация не найдена."]
            else:
                context_list = context_docs
                logger.info("✅ Контекст найден", docs_count=len(context_docs))

            # Создаем RAG промпт
            rag_prompt = await self._prompt_service.create_rag_prompt(
                query=user_query,
                context=context_list
            )

            logger.debug("Современный RAG промпт создан",
                        original_len=len(user_query),
                        rag_len=len(rag_prompt),
                        context_docs=len(context_list))

            return rag_prompt

        except Exception as e:
            logger.error("❌ Ошибка создания современного RAG промпта", error=str(e), exc_info=True)
            # Fallback: возвращаем промпт с пустым контекстом
            return await self._prompt_service.create_rag_prompt(
                query=user_query,
                context=["Ошибка поиска в документации."]
            )

    async def search_relevant_context(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> list[str]:
        """
        Найти релевантный контекст для запроса с современными алгоритмами поиска.

        Args:
            query: Поисковый запрос
            k: Количество документов для поиска
            min_score: Минимальный score для фильтрации результатов

        Returns:
            Список релевантных документов
        """
        await self.initialize()

        logger.info("🔍 Современный RAG поиск контекста",
                   query_preview=query[:100],
                   k=k,
                   min_score=min_score)

        try:
            # Создаем эмбеддинг запроса
            logger.debug("🧠 Создание эмбеддинга для поиска")
            query_embedding = await self._embedding_service.embed_text(query)
            logger.debug("Эмбеддинг создан", embedding_dim=len(query_embedding))

            # Выполняем поиск в векторном хранилище
            search_results = await self._vector_store.search(
                query_embedding=query_embedding,
                k=k,
                min_score=min_score
            )
            logger.info("Результаты поиска получены", found_docs=len(search_results))

            if not search_results:
                logger.warning("❌ Не найдено релевантных документов")
                return []

            # Обрабатываем результаты с учетом лимитов
            context_docs = []
            total_length = 0
            unique_sources = set()

            for i, result in enumerate(search_results):
                document = result.document
                score = result.score
                content = document.content
                metadata = document.metadata

                # Извлекаем метаданные
                title = metadata.get('title', 'Без названия')
                source_url = metadata.get('source', '')
                chunking_strategy = metadata.get('chunking_strategy', 'unknown')
                embedding_provider = metadata.get('embedding_provider', 'unknown')

                # Собираем статистику источников
                if source_url:
                    unique_sources.add((title, source_url))

                # Проверяем лимит длины контекста
                if total_length + len(content) > self._max_context_length:
                    remaining = self._max_context_length - total_length
                    if remaining > 100:  # Минимальный размер для полезного контента
                        truncated_content = content[:remaining-50] + "...[ОБРЕЗАНО]"
                        context_docs.append(truncated_content)
                        total_length += len(truncated_content)
                        logger.debug(f"Документ {i+1} обрезан до {remaining} символов")
                    break

                context_docs.append(content)
                total_length += len(content)

                logger.debug(f"Документ {i+1} добавлен",
                           score=f"{score:.3f}",
                           title=title,
                           length=len(content),
                           chunking_strategy=chunking_strategy,
                           embedding_provider=embedding_provider)

            # Логируем финальную статистику
            sources_list = [f"{title} - {url}" for title, url in unique_sources]

            logger.info("✅ Современный RAG контекст готов",
                       documents=len(context_docs),
                       total_chars=total_length,
                       unique_sources=len(sources_list),
                       avg_score=f"{sum(r.score for r in search_results[:len(context_docs)]) / len(context_docs):.3f}",
                       sources=sources_list[:3])  # Показываем только первые 3 источника

            return context_docs

        except Exception as e:
            logger.error("❌ Ошибка современного RAG поиска", error=str(e), exc_info=True)
            return []

    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any]
    ) -> bool:
        """
        Добавить документ в векторное хранилище.

        Args:
            content: Содержимое документа
            metadata: Метаданные документа

        Returns:
            True если документ успешно добавлен
        """
        await self.initialize()

        try:
            # Создаем эмбеддинг
            embedding = await self._embedding_service.embed_text(content)

            # Создаем документ
            document = Document(
                id=metadata.get('id', str(hash(content))),
                content=content,
                metadata=metadata
            )

            # Добавляем в векторное хранилище
            await self._vector_store.add_document(document, embedding)

            logger.info("✅ Документ добавлен",
                       title=metadata.get('title', 'Без названия'),
                       content_length=len(content))
            return True

        except Exception as e:
            logger.error("❌ Ошибка добавления документа", error=str(e), exc_info=True)
            return False

    async def batch_add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 32
    ) -> dict[str, Any]:
        """
        Добавить документы батчами.

        Args:
            documents: Список документов [{"content": str, "metadata": dict}]
            batch_size: Размер батча

        Returns:
            Статистика добавления
        """
        await self.initialize()

        logger.info("📦 Начинаем батчевое добавление документов",
                   total_docs=len(documents),
                   batch_size=batch_size)

        if not documents:
            return {"success": False, "error": "Нет документов для добавления"}

        try:
            stats = {
                "total_docs": len(documents),
                "processed_docs": 0,
                "failed_docs": 0,
                "batches": 0,
                "success": True
            }

            # Обрабатываем батчами
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1

                logger.info(f"📦 Обрабатываем батч {batch_num}", batch_size=len(batch))

                # Создаем эмбеддинги для батча
                texts = [doc["content"] for doc in batch]
                embeddings = await self._embedding_service.embed_batch(texts)

                # Добавляем документы
                for doc_data, embedding in zip(batch, embeddings, strict=False):
                    try:
                        document = Document(
                            id=doc_data["metadata"].get('id', str(hash(doc_data["content"]))),
                            content=doc_data["content"],
                            metadata=doc_data["metadata"]
                        )

                        await self._vector_store.add_document(document, embedding)
                        stats["processed_docs"] += 1

                    except Exception as e:
                        logger.error("❌ Ошибка добавления документа в батче", error=str(e))
                        stats["failed_docs"] += 1

                stats["batches"] += 1
                logger.info(f"✅ Батч {batch_num} обработан",
                           processed=stats["processed_docs"],
                           failed=stats["failed_docs"])

            logger.info("✅ Батчевое добавление завершено", **stats)
            return stats

        except Exception as e:
            logger.error("❌ Ошибка батчевого добавления", error=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_collection_stats(self) -> dict[str, Any]:
        """Получить статистику коллекции."""
        await self.initialize()

        try:
            stats = await self._vector_store.get_stats()
            logger.info("📊 Статистика коллекции получена", **stats)
            return stats
        except Exception as e:
            logger.error("❌ Ошибка получения статистики", error=str(e), exc_info=True)
            return {"error": str(e)}

    async def is_ready(self) -> bool:
        """Проверить готовность сервиса."""
        try:
            await self.initialize()

            # Проверяем компоненты
            embedding_ready = await self._embedding_service.is_ready()
            vector_store_ready = await self._vector_store.is_ready()

            ready = embedding_ready and vector_store_ready

            logger.info("🔍 Проверка готовности современного RAG сервиса",
                       embedding_ready=embedding_ready,
                       vector_store_ready=vector_store_ready,
                       overall_ready=ready)

            return ready

        except Exception as e:
            logger.error("❌ Ошибка проверки готовности", error=str(e), exc_info=True)
            return False

    # Методы для обратной совместимости со старым API
    async def enhance_prompt_with_context(
        self,
        original_prompt: str,
        context: list[str]
    ) -> str:
        """Улучшить промпт контекстной информацией (совместимость)."""
        if not context:
            logger.debug("Нет контекста для улучшения промпта")
            return original_prompt

        # Используем новый метод создания RAG промпта
        enhanced_prompt = await self._prompt_service.create_rag_prompt(
            query=original_prompt,
            context=context
        )

        logger.debug("Промпт улучшен современным методом",
                    original_len=len(original_prompt),
                    enhanced_len=len(enhanced_prompt),
                    context_docs=len(context))

        return enhanced_prompt
