"""Современный модуль для построения RAG векторной базы."""

import json
import uuid
from pathlib import Path
from typing import Any

import structlog

from ..config.settings import get_settings
from .modern_chunker import ChunkingStrategy, ModernSemanticChunker
from .modern_embeddings import EmbeddingProvider, ModernEmbeddingService
from .modern_vector_store import ModernVectorStoreFactory, VectorStoreType
from .protocols import Document

logger = structlog.get_logger(__name__)


def preprocess_text(text: str) -> str:
    """Предобработка текста."""
    if not text:
        return ""

    import re
    # Убираем лишние символы форматирования
    text = text.replace("false", "").replace("true", "")
    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


class ModernRAGBuilder:
    """Современный построитель RAG векторной базы с лучшими практиками 2025."""

    def __init__(
        self,
        # Chunking settings
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        chunk_size: int = 512,
        overlap_size: int = 50,

        # Embedding settings
        embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
        embedding_model: str = "BAAI/bge-m3",

        # Vector store settings
        vector_store_type: VectorStoreType = VectorStoreType.CHROMADB,
        collection_name: str = "rag_documents",

        # Processing settings
        batch_size: int = 32,
        test_mode: bool = False
    ):
        """
        Инициализация современного RAG билдера.

        Args:
            chunking_strategy: Стратегия чанкинга (semantic, title, hybrid)
            chunk_size: Размер чанка
            overlap_size: Размер перекрытия
            embedding_provider: Провайдер эмбеддингов
            embedding_model: Модель эмбеддингов
            vector_store_type: Тип векторного хранилища
            collection_name: Имя коллекции
            batch_size: Размер батча
            test_mode: Режим тестирования
        """
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.test_mode = test_mode

        # Получаем настройки
        self.settings = get_settings()

        # Инициализируем компоненты
        self.chunker: ModernSemanticChunker | None = None
        self.embedding_service: ModernEmbeddingService | None = None
        self.vector_store = None

    async def _initialize_components(self) -> None:
        """Инициализация современных компонентов."""
        logger.info("🚀 Инициализация современных RAG компонентов")

        # Инициализируем чанкер
        self.chunker = ModernSemanticChunker(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            overlap=self.overlap_size
        )
        await self.chunker.initialize()
        logger.info("✅ Современный семантический чанкер готов", strategy=self.chunking_strategy.value)

        # Инициализируем эмбеддинг сервис
        embedding_config = {
            "provider_type": self.embedding_provider.value,
            "model_name": self.embedding_model
        }

        # Добавляем специфичные настройки для провайдеров
        if self.embedding_provider == EmbeddingProvider.MODEL2VEC:
            embedding_config["model_name"] = "minishlab/potion-base-8M"  # Быстрая модель
        elif self.embedding_provider == EmbeddingProvider.OPENAI:
            embedding_config["api_key"] = self.settings.openai_api_key

        self.embedding_service = ModernEmbeddingService(**embedding_config)
        await self.embedding_service.initialize()
        logger.info("✅ Современный эмбеддинг сервис готов",
                   provider=self.embedding_provider.value,
                   model=self.embedding_model)

        # Создаем vector store
        store_config = {
            "collection_name": self.collection_name,
            "persist_path": getattr(self.settings, 'chromadb_path', './data/chromadb')
        }
        
        self.vector_store = ModernVectorStoreFactory.create_store("chroma", **store_config)
        logger.info("✅ Векторное хранилище создано", type="ChromaDB")

    async def build_from_confluence_data(
        self,
        input_file: Path,
        output_dir: Path | None = None
    ) -> dict[str, Any]:
        """
        Построить современную RAG базу из данных Confluence.

        Args:
            input_file: Файл с данными Confluence
            output_dir: Выходная директория (для совместимости, не используется в modern режиме)

        Returns:
            Словарь со статистикой построения
        """
        logger.info("🚀 Начинаем построение современной RAG векторной базы")

        # Инициализируем компоненты
        await self._initialize_components()

        # Загружаем данные
        logger.info("📖 Загружаем данные из Confluence", file=str(input_file))
        with open(input_file, encoding='utf-8') as f:
            confluence_data = json.load(f)

        if self.test_mode and len(confluence_data) > 5:
            confluence_data = confluence_data[:5]
            logger.info("🧪 Тестовый режим: ограничиваем до 5 документов")

        logger.info("📊 Данные загружены", total_pages=len(confluence_data))

        # Обрабатываем документы
        documents = await self._process_documents(confluence_data)
        logger.info("📄 Документы обработаны", total_chunks=len(documents))

        # Создаем эмбеддинги и сохраняем в векторное хранилище
        stats = await self._create_embeddings_and_store(documents)

        # Тестируем поиск
        if self.test_mode:
            await self._test_modern_search()

        logger.info("🎉 Современная RAG база успешно создана!")
        return stats

    async def _process_documents(self, confluence_data: list[dict[str, Any]]) -> list[Document]:
        """Обработать документы из Confluence современными методами."""
        documents = []

        for page in confluence_data:
            content = page.get("page_content", "")
            metadata = page.get("metadata", {})

            # Предобрабатываем текст
            content = preprocess_text(content)
            if not content or len(content) < 50:
                logger.debug("Пропускаем короткий документ", title=metadata.get("title", ""))
                continue

            title = metadata.get("title", "Без названия")
            source = metadata.get("source", "")
            page_id = metadata.get("id", "")

            # Базовые метаданные
            base_metadata = {
                "title": title,
                "source": source,
                "page_id": page_id,
                "when": metadata.get("when", ""),
                "chunking_strategy": self.chunking_strategy.value,
                "embedding_provider": self.embedding_provider.value
            }

            # Современное чанкинг с метаданными
            logger.debug("Обрабатываем документ современным чанкером",
                        title=title,
                        strategy=self.chunking_strategy.value)

            chunks_with_metadata = await self.chunker.chunk_document(
                content=content,
                metadata=base_metadata
            )

            # Конвертируем в Document объекты
            for chunk_data in chunks_with_metadata:
                doc = Document(
                    id=str(uuid.uuid4()),
                    content=chunk_data["content"],
                    metadata=chunk_data["metadata"]
                )
                documents.append(doc)

            logger.debug("Документ обработан",
                        title=title,
                        original_length=len(content),
                        chunks_count=len(chunks_with_metadata))

        return documents

    async def _create_embeddings_and_store(self, documents: list[Document]) -> dict[str, Any]:
        """Создать эмбеддинги и сохранить в современное векторное хранилище."""
        logger.info("🧠 Создание эмбеддингов и сохранение в векторное хранилище",
                   total_docs=len(documents))

        # Обрабатываем батчами для эффективности
        batch_stats = []
        total_processed = 0

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

            logger.info(f"📦 Обрабатываем батч {batch_num}/{total_batches}",
                       batch_size=len(batch))

            # Создаем эмбеддинги
            texts = [doc.content for doc in batch]
            embeddings = await self.embedding_service.embed_batch(texts)

            # Добавляем эмбеддинги к документам
            for doc, embedding in zip(batch, embeddings, strict=False):
                doc.embedding = embedding

            # Сохраняем батч в векторное хранилище
            await self.vector_store.add_documents(batch)

            total_processed += len(batch)
            batch_stats.append({
                "batch_num": batch_num,
                "docs_processed": len(batch),
                "total_processed": total_processed
            })

            logger.info(f"✅ Батч {batch_num} обработан",
                       processed=len(batch),
                       total_progress=f"{total_processed}/{len(documents)}")

        # Финальная статистика
        stats = {
            "total_documents": len(documents),
            "total_batches": len(batch_stats),
            "chunking_strategy": self.chunking_strategy.value,
            "embedding_provider": self.embedding_provider.value,
            "vector_store_type": self.vector_store_type.value,
            "collection_name": self.collection_name,
            "success": True
        }

        logger.info("✅ Эмбеддинги созданы и сохранены", **stats)
        return stats

    async def _test_modern_search(self) -> None:
        """Протестировать современный поиск."""
        logger.info("🔍 Начинаем тестирование современного поиска")

        test_queries = [
            "Как установить Jenkins?",
            "Основные команды Docker",
            "Что такое Kubernetes?",
            "Git команды для работы с ветками",
            "HTTP коды ответов API"
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"Запрос {i}: {query}")

            # Создаем эмбеддинг запроса
            query_embedding = await self.embedding_service.embed_text(query)

            # Выполняем поиск
            results = await self.vector_store.search(query_embedding, k=3)

            if results:
                logger.info(f"✅ Найдено {len(results)} релевантных документов")
                for j, result in enumerate(results, 1):
                    doc = result.document
                    score = result.score
                    title = doc.metadata.get('title', 'Без названия')
                    preview = doc.content[:150].replace('\n', ' ') + "..." if len(doc.content) > 150 else doc.content
                    logger.info(f"  Документ {j} (score: {score:.3f}): {title}")
                    logger.info(f"    {preview}")
            else:
                logger.warning(f"❌ Документы не найдены для запроса: {query}")


        logger.info("✅ Тестирование современного поиска завершено")


async def build_modern_rag_from_confluence(
    input_file: str = "data/confluence_data.json",
    output_dir: str = "data/modern_rag_index",
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    vector_store_type: VectorStoreType = VectorStoreType.CHROMADB,
    test_mode: bool = False
) -> dict[str, Any]:
    """
    Построить современную RAG базу из данных Confluence.

    Эта функция обеспечивает обратную совместимость с существующим API.
    """
    builder = ModernRAGBuilder(
        chunking_strategy=chunking_strategy,
        embedding_provider=embedding_provider,
        vector_store_type=vector_store_type,
        test_mode=test_mode
    )

    return await builder.build_from_confluence_data(
        input_file=Path(input_file),
        output_dir=Path(output_dir) if output_dir else None
    )
