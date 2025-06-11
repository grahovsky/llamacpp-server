#!/usr/bin/env python3
"""Современный скрипт для создания RAG системы с передовыми компонентами 2025."""

import asyncio
import argparse
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append('.')

import structlog

from llamacpp_server.retrieval.modern_chunker import ChunkingStrategy
from llamacpp_server.retrieval.modern_embeddings import EmbeddingProvider
from llamacpp_server.retrieval.modern_vector_store import VectorStoreType
from llamacpp_server.retrieval.modern_rag_service import ModernRAGService
from llamacpp_server.config.settings import get_settings

# Настройка логирования
logger = structlog.get_logger(__name__)


async def build_rag_database(
    data_path: Path,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    embedding_model: str = "BAAI/bge-m3",
    collection_name: str = "rag_documents",
    min_score: float = 0.1  # Минимальный порог релевантности
) -> dict:
    """
    Построить современную RAG базу данных.
    
    Args:
        data_path: Путь к данным
        chunking_strategy: Стратегия чанкинга
        embedding_provider: Провайдер эмбеддингов
        embedding_model: Модель эмбеддингов
        collection_name: Имя коллекции
        min_score: Минимальный score для тестирования
        
    Returns:
        Статистика создания базы
    """
    
    logger.info("🚀 Создание современной RAG системы",
               chunking_strategy=chunking_strategy.value,
               embedding_provider=embedding_provider.value,
               embedding_model=embedding_model,
               collection_name=collection_name)
    
    # Получаем настройки
    settings = get_settings()
    
    # Создаем RAG сервис (всегда ChromaDB)
    rag_service = ModernRAGService(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_store_type=VectorStoreType.CHROMADB,
        collection_name=collection_name,
        settings=settings
    )
    
    await rag_service.initialize()
    
    # Читаем данные из JSON файла
    import json
    
    with open(data_path, 'r', encoding='utf-8') as f:
        confluence_data = json.load(f)
    
    logger.info("📂 Данные загружены", 
               pages_count=len(confluence_data),
               source_file=str(data_path))
    
    # Создаем chunker
    from llamacpp_server.retrieval.modern_chunker import ModernChunkerFactory
    
    chunker = ModernChunkerFactory.create_chunker(
        strategy=chunking_strategy,
        chunk_size=512,
        overlap=50
    )
    
    # Обрабатываем страницы
    all_documents = []
    
    for page in confluence_data:
        try:
            # Создаем базовые метаданные
            metadata = {
                "title": page.get("title", "Без названия"),
                "source": page.get("url", ""),
                "page_id": page.get("id", ""),
                "when": page.get("when", ""),
                "chunking_strategy": chunking_strategy.value,
                "embedding_provider": embedding_provider.value
            }
            
            # Чанкинг контента
            content = page.get("body", "")
            if not content.strip():
                continue
                
            chunks = await chunker.chunk_text(content, metadata)
            
            # Готовим документы для добавления
            for chunk in chunks:
                doc_data = {
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                all_documents.append(doc_data)
                
        except Exception as e:
            logger.warning("Ошибка обработки страницы", 
                         title=page.get("title", "Unknown"),
                         error=str(e))
            continue
    
    logger.info("📝 Документы подготовлены", 
               total_documents=len(all_documents))
    
    # Добавляем документы батчами
    stats = await rag_service.batch_add_documents(
        documents=all_documents,
        batch_size=32
    )
    
    # Тестируем поиск
    test_query = "настройка конфигурации системы"
    logger.info("🔍 Тестирование поиска", query=test_query)
    
    test_results = await rag_service.search_relevant_context(
        query=test_query,
        k=3,
        min_score=min_score
    )
    
    logger.info("✅ Тест поиска завершен", 
               found_docs=len(test_results),
               min_score=min_score)
    
    # Финальная статистика
    collection_stats = await rag_service.get_collection_stats()
    
    final_stats = {
        **stats,
        "test_results": len(test_results),
        "collection_stats": collection_stats,
        "chunking_strategy": chunking_strategy.value,
        "embedding_provider": embedding_provider.value,
        "vector_store": "ChromaDB"
    }
    
    logger.info("🎯 RAG система создана успешно", **final_stats)
    
    return final_stats


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Создание современной RAG системы с ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python modern_build_rag.py data/confluence.json
  python modern_build_rag.py data/confluence.json --semantic --openai
  python modern_build_rag.py data/confluence.json --hybrid --model2vec
        """
    )
    
    parser.add_argument(
        "data_path",
        type=Path,
        help="Путь к JSON файлу с данными Confluence"
    )
    
    # Стратегия чанкинга
    chunking_group = parser.add_mutually_exclusive_group()
    chunking_group.add_argument("--semantic", action="store_true", help="Семантический чанкинг")
    chunking_group.add_argument("--title", action="store_true", help="Чанкинг по заголовкам")
    chunking_group.add_argument("--hybrid", action="store_true", help="Гибридный чанкинг (по умолчанию)")
    
    # Провайдер эмбеддингов
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument("--sentence-transformers", action="store_true", help="SentenceTransformers (по умолчанию)")
    embedding_group.add_argument("--model2vec", action="store_true", help="Model2Vec")
    embedding_group.add_argument("--openai", action="store_true", help="OpenAI")
    
    parser.add_argument("--collection", default="rag_documents", help="Имя коллекции")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Модель эмбеддингов")
    parser.add_argument("--min-score", type=float, default=0.1, help="Минимальный score для тестирования")
    
    return parser.parse_args()


async def main():
    """Главная функция."""
    args = parse_arguments()
    
    # Определяем стратегию чанкинга
    if args.semantic:
        chunking_strategy = ChunkingStrategy.SEMANTIC
    elif args.title:
        chunking_strategy = ChunkingStrategy.TITLE
    else:
        chunking_strategy = ChunkingStrategy.HYBRID
    
    # Определяем провайдер эмбеддингов
    if args.model2vec:
        embedding_provider = EmbeddingProvider.MODEL2VEC
    elif args.openai:
        embedding_provider = EmbeddingProvider.OPENAI
    else:
        embedding_provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    
    # Проверяем существование файла
    if not args.data_path.exists():
        logger.error("❌ Файл не найден", path=str(args.data_path))
        return 1
    
    logger.info("🎯 Параметры сборки",
               data_path=str(args.data_path),
               chunking_strategy=chunking_strategy.value,
               embedding_provider=embedding_provider.value,
               embedding_model=args.model,
               collection_name=args.collection,
               vector_store="ChromaDB")
    
    try:
        stats = await build_rag_database(
            data_path=args.data_path,
            chunking_strategy=chunking_strategy,
            embedding_provider=embedding_provider,
            embedding_model=args.model,
            collection_name=args.collection,
            min_score=args.min_score
        )
        
        logger.info("✅ Современная RAG система готова!", **stats)
        return 0
        
    except Exception as e:
        logger.error("❌ Ошибка создания RAG системы", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    # Настройка structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    exit_code = asyncio.run(main()) 