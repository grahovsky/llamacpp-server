#!/usr/bin/env python3
"""Тест семантической разбивки текста."""

import asyncio

import structlog
from sentence_transformers import SentenceTransformer

from llamacpp_server.retrieval.semantic_chunker import SemanticChunker

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


async def test_semantic_chunking():
    """Тестирование семантической разбивки."""
    logger.info("🧪 Начинаем тест семантической разбивки")

    # Тестовый текст с URL и email
    test_text = """
    Jenkins - это система автоматизации, которая помогает автоматизировать части процесса разработки программного обеспечения, связанные с созданием, тестированием и развертыванием. Подробности можно найти на https://jenkins.io и связаться с нами по admin@company.com.
    
    Установка Jenkins довольно проста. Сначала вам нужно скачать Jenkins с официального сайта. Затем запустите установочный файл и следуйте инструкциям мастера установки.
    
    Настройка Jenkins включает в себя несколько важных шагов. Во-первых, необходимо настроить безопасность. Во-вторых, установить необходимые плагины. В-третьих, создать первое задание.
    
    Плагины Jenkins расширяют функциональность системы. Существует огромное количество плагинов для различных целей: интеграция с системами контроля версий, уведомления, развертывание и многое другое. Список плагинов доступен на https://plugins.jenkins.io.
    
    Docker является платформой для разработки, доставки и запуска приложений. Docker использует контейнеризацию для упаковки приложения и его зависимостей в легковесный, портативный контейнер. Официальная документация находится на https://docs.docker.com.
    
    Kubernetes - это платформа оркестрации контейнеров с открытым исходным кодом. Она автоматизирует развертывание, масштабирование и управление контейнеризованными приложениями. Больше информации на https://kubernetes.io или support@kubernetes.io.
    """

    # Загружаем модель
    logger.info("🧠 Загружаем модель эмбеддингов")
    model = SentenceTransformer("BAAI/bge-m3")

    # Создаем семантический разбивщик
    chunker = SemanticChunker(
        embedding_model=model,
        max_chunk_size=800,
        min_chunk_size=50,
        similarity_threshold=0.6,
        overlap_sentences=1
    )

    # Тестируем разбивку
    logger.info("🔪 Выполняем семантическую разбивку")
    chunks = await chunker.chunk_text(
        text=test_text.strip(),
        metadata={"title": "Test Document", "source": "https://test.com"}
    )

    # Выводим результаты
    logger.info("📊 Результаты разбивки", total_chunks=len(chunks))

    for i, chunk in enumerate(chunks):
        content = chunk["content"]
        metadata = chunk["metadata"]

        print(f"\n{'='*60}")
        print(f"ЧАНК {i+1}/{len(chunks)}")
        print(f"{'='*60}")
        print(f"Размер: {metadata.get('chunk_size', len(content))} символов")
        print(f"Тип: {metadata.get('chunk_type', 'unknown')}")
        print(f"Индекс: {metadata.get('chunk_index', i)}")
        print("\nСодержимое:")
        print("-" * 40)
        print(content)
        print("-" * 40)

    # Проверим защиту URL и email
    logger.info("🔒 Проверяем защиту URL и email")

    urls_found = []
    emails_found = []

    for chunk in chunks:
        content = chunk["content"]
        # Простая проверка наличия URL и email
        if "http" in content or "www." in content:
            urls_found.append(chunk["metadata"]["chunk_index"])
        if "@" in content:
            emails_found.append(chunk["metadata"]["chunk_index"])

    logger.info("✅ Анализ защиты завершен",
               chunks_with_urls=urls_found,
               chunks_with_emails=emails_found)

    print(f"\n{'='*60}")
    print("ИТОГИ ТЕСТА")
    print(f"{'='*60}")
    print(f"Исходный текст: {len(test_text)} символов")
    print(f"Получено чанков: {len(chunks)}")
    print(f"Средний размер чанка: {sum(len(c['content']) for c in chunks) // len(chunks)} символов")
    print(f"Чанки с URL: {len(urls_found)}")
    print(f"Чанки с email: {len(emails_found)}")
    print("Тип разбивки: семантическая")


if __name__ == "__main__":
    asyncio.run(test_semantic_chunking())
