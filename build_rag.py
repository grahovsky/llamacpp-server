#!/usr/bin/env python3
"""Построение RAG векторной базы из confluence_data.json с семантической разбивкой."""

import asyncio
import json
from pathlib import Path

import structlog

from llamacpp_server.retrieval.rag_builder import RAGBuilder

# Настраиваем логирование
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

async def test_rag_search(index_dir: Path) -> None:
    """Протестировать поиск в созданном RAG индексе."""
    from llamacpp_server.retrieval.rag_service import RAGService
    from llamacpp_server.retrieval.embedding_service import EmbeddingService  
    from llamacpp_server.retrieval.faiss_store import FaissVectorStore
    
    logger.info("🔍 Начинаем тестирование поиска в RAG")
    
    try:
        # Инициализируем сервисы
        embedding_service = EmbeddingService()
        await embedding_service.is_ready()
        
        vector_store = FaissVectorStore()
        await vector_store.load_index(str(index_dir))
        
        rag_service = RAGService(embedding_service, vector_store)
        
        # Тестовые запросы
        test_queries = [
            "Как установить Jenkins?",
            "Основные команды Docker",
            "Что такое Kubernetes?", 
            "Git команды для работы с ветками",
            "HTTP коды ответов API"
        ]
        
        logger.info("🎯 Выполняем тестовые запросы")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Запрос {i}: {query}")
            
            # Выполняем поиск
            context = await rag_service.search_relevant_context(query, k=3)
            
            if context:
                logger.info(f"✅ Найдено {len(context)} релевантных документов")
                for j, doc in enumerate(context, 1):
                    preview = doc[:150].replace('\n', ' ') + "..." if len(doc) > 150 else doc
                    logger.info(f"  Документ {j}: {preview}")
            else:
                logger.warning(f"❌ Документы не найдены для запроса: {query}")
            
            print("-" * 80)
            
        logger.info("✅ Тестирование поиска завершено")
        
    except Exception as e:
        logger.error("❌ Ошибка при тестировании поиска", error=str(e), exc_info=True)


async def main():
    """Главная функция для построения RAG с семантической разбивкой."""
    import sys

    # Определяем режим работы из аргументов командной строки
    mode = "semantic"  # По умолчанию семантическая разбивка
    test_mode = False
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--simple":
            mode = "simple"
        elif arg == "--semantic":
            mode = "semantic"
        elif arg == "--test":
            mode = "semantic"
            test_mode = True
        elif arg == "--help":
            print("Использование:")
            print("  python build_rag.py --semantic   # Семантическая разбивка (по умолчанию)")
            print("  python build_rag.py --simple     # Простая разбивка с перекрытием")
            print("  python build_rag.py --test       # Быстрое тестирование на sample данных")
            print("  python build_rag.py --help       # Показать справку")
            return

    use_semantic = mode == "semantic"
    chunking_type = "семантической" if use_semantic else "простой"
    
    if test_mode:
        logger.info("🧪 Режим быстрого тестирования RAG")
        
        # Настройки для тестирования
        input_file = Path("data/confluence_test_data.json")
        output_dir = Path("data/faiss_test_index")
        
        # Создаем RAG Builder с ускоренными настройками для тестирования
        rag_builder = RAGBuilder(
            model_name="BAAI/bge-m3",
            embedding_dim=1024,
            chunk_size=800,  # Меньший размер для тестирования
            overlap_size=100,
            batch_size=16,   # Меньший batch для экономии памяти
            use_semantic_chunking=True,  # Всегда используем семантическую разбивку в тесте
            similarity_threshold=0.6     # Немного ниже порог для большего количества границ
        )
        
        logger.info("🚀 Начинаем быстрое построение тестового RAG")
        
    else:
        logger.info(f"🚀 Начинаем построение RAG векторной базы с {chunking_type} разбивкой")
        
        # Настройки для полной сборки
        input_file = Path("data/confluence_data.json")
        output_dir = Path("data/faiss_index")
        
        # Создаем RAG Builder с выбранным типом разбивки
        rag_builder = RAGBuilder(
            model_name="BAAI/bge-m3",
            embedding_dim=1024,
            chunk_size=512,
            overlap_size=50,
            batch_size=100,
            use_semantic_chunking=use_semantic,
            similarity_threshold=0.7  # Порог сходства для разделения чанков
        )

    # Строим RAG базу
    await rag_builder.build_from_confluence_data(input_file, output_dir)

    if test_mode:
        logger.info("🎉 Тестовый RAG успешно создан!")
        
        # Тестируем поиск
        await test_rag_search(output_dir)
        
        logger.info("📋 Особенности тестового режима:")
        logger.info("  ✅ Быстрая сборка на 5 sample документах")
        logger.info("  ✅ Семантическая разбивка с оптимизированными параметрами")
        logger.info("  ✅ Автоматическое тестирование поиска")
        logger.info("  ✅ Детальная диагностика качества")
        
    else:
        logger.info(f"🎉 RAG с {chunking_type} разбивкой успешно создан!")

        if use_semantic:
            logger.info("📋 Особенности семантической разбивки:")
            logger.info("  ✅ Семантическая разбивка с защитой URL и email")
            logger.info("  ✅ Анализ сходства предложений с адаптивным порогом")
            logger.info("  ✅ Автоматические границы по смыслу")
            logger.info("  ✅ Умное семантическое перекрытие")
            logger.info("  ✅ Иерархическая разбивка больших документов")
            logger.info("  ✅ Скользящее окно для анализа связности")
        else:
            logger.info("📋 Особенности простой разбивки:")
            logger.info("  ✅ Фиксированный размер чанков")
            logger.info("  ✅ Перекрытие по символам")
            logger.info("  ✅ Быстрая обработка")

        logger.info("🔍 Общие улучшения:")
        logger.info("  ✅ Улучшенное форматирование контекста с эмодзи")
        logger.info("  ✅ Метаданные с информацией о типе разбивки")
        logger.info("  ✅ Отображение релевантности и источников в поиске")
        logger.info("  ✅ Анализ качества разбивки с рекомендациями")


if __name__ == "__main__":
    asyncio.run(main())
