#!/usr/bin/env python3
"""Построение RAG векторной базы из confluence_data.json с семантической разбивкой."""

import asyncio
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


async def main():
    """Главная функция для построения RAG с семантической разбивкой."""
    import sys

    # Определяем тип разбивки из аргументов командной строки
    use_semantic = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "--simple":
            use_semantic = False
        elif sys.argv[1] == "--semantic":
            use_semantic = True
        elif sys.argv[1] == "--help":
            print("Использование:")
            print("  python build_rag.py --semantic   # Семантическая разбивка (по умолчанию)")
            print("  python build_rag.py --simple     # Простая разбивка с перекрытием")
            print("  python build_rag.py --help       # Показать справку")
            return

    chunking_type = "семантической" if use_semantic else "простой"
    logger.info(f"🚀 Начинаем построение RAG векторной базы с {chunking_type} разбивкой")

    # Настройки
    input_file = Path("data/confluence_data.json")
    output_dir = Path("data/faiss_index")

    # Создаем RAG Builder с выбранным типом разбивки
    rag_builder = RAGBuilder(
        model_name="BAAI/bge-m3",
        embedding_dim=1024,
        chunk_size=1500,
        overlap_size=200,
        batch_size=32,
        use_semantic_chunking=use_semantic,
        similarity_threshold=0.7  # Порог сходства для разделения чанков
    )

    # Строим RAG базу
    await rag_builder.build_from_confluence_data(input_file, output_dir)

    logger.info(f"🎉 RAG с {chunking_type} разбивкой успешно создан!")

    if use_semantic:
        logger.info("📋 Особенности семантической разбивки:")
        logger.info("  ✅ Семантическая разбивка с защитой URL и email")
        logger.info("  ✅ Анализ сходства предложений")
        logger.info("  ✅ Автоматические границы по смыслу")
        logger.info("  ✅ Умное перекрытие предложений")
    else:
        logger.info("📋 Особенности простой разбивки:")
        logger.info("  ✅ Фиксированный размер чанков")
        logger.info("  ✅ Перекрытие по символам")
        logger.info("  ✅ Быстрая обработка")

    logger.info("🔍 Общие улучшения:")
    logger.info("  ✅ Улучшенное форматирование контекста с эмодзи")
    logger.info("  ✅ Метаданные с информацией о типе разбивки")
    logger.info("  ✅ Отображение релевантности и источников в поиске")


if __name__ == "__main__":
    asyncio.run(main())
