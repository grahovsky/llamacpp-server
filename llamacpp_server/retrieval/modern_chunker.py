"""Современная замена для семантического чанкинга с использованием chonkie и встроенных алгоритмов."""

import re
import uuid
from enum import Enum
from typing import Any

import structlog
from chonkie import SentenceChunker, TokenChunker, RecursiveChunker

logger = structlog.get_logger(__name__)


class ChunkingStrategy(Enum):
    """Стратегии чанкинга."""
    SEMANTIC = "semantic"
    TITLE_BASED = "title"
    HYBRID = "hybrid"


class ModernSemanticChunker:
    """Современный семантический чанкер с использованием chonkie и встроенных алгоритмов."""

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7
    ):
        """
        Инициализация современного чанкера.

        Args:
            strategy: Стратегия чанкинга
            chunk_size: Максимальный размер чанка
            overlap: Размер перекрытия
            min_chunk_size: Минимальный размер чанка
            similarity_threshold: Порог семантической схожести
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold

        # Инициализируем чанкеры из chonkie
        self.token_chunker = TokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        self.recursive_chunker = RecursiveChunker(
            chunk_size=chunk_size
        )

        self.sentence_chunker = SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        self.initialized = False

        logger.info("Современный чанкер инициализирован",
                   strategy=strategy.value,
                   chunk_size=chunk_size,
                   overlap=overlap)

    async def initialize(self) -> None:
        """Асинхронная инициализация."""
        if self.initialized:
            return

        logger.info("🚀 Инициализация современного чанкера")

        # Здесь могут быть дополнительные инициализации
        # Например, загрузка моделей для семантического анализа

        self.initialized = True
        logger.info("✅ Современный чанкер готов")

    async def chunk_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Чанкинг документа с выбранной стратегией.

        Args:
            content: Содержимое документа
            metadata: Метаданные документа

        Returns:
            Список чанков с метаданными
        """
        await self.initialize()

        if not content or len(content) < self.min_chunk_size:
            logger.warning("Документ слишком короткий для чанкинга", length=len(content))
            return []

        logger.info("Начинаем чанкинг документа",
                   strategy=self.strategy.value,
                   content_length=len(content))

        if self.strategy == ChunkingStrategy.SEMANTIC:
            return await self._chunk_semantic(content, metadata)
        elif self.strategy == ChunkingStrategy.TITLE_BASED:
            return await self._chunk_by_title(content, metadata)
        elif self.strategy == ChunkingStrategy.HYBRID:
            return await self._chunk_hybrid(content, metadata)
        else:
            raise ValueError(f"Неизвестная стратегия чанкинга: {self.strategy}")

    async def _chunk_semantic(
        self,
        content: str,
        metadata: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Семантический чанкинг с использованием sentence-based подхода."""

        logger.debug("Используем семантический чанкинг")

        # Используем sentence chunker из chonkie как основу
        sentence_chunks = self.sentence_chunker.chunk(content)

        # Применяем дополнительную семантическую логику
        semantic_chunks = await self._apply_semantic_logic(sentence_chunks)

        # Форматируем результат
        result = []
        for i, chunk_content in enumerate(semantic_chunks):
            if len(chunk_content.strip()) < self.min_chunk_size:
                continue

            chunk_metadata = {
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "total_chunks": len(semantic_chunks),
                "chunking_strategy": "semantic",
                "chunk_type": "semantic_sentence",
                "character_count": len(chunk_content),
                "estimated_tokens": self._estimate_tokens(chunk_content),
                **(metadata or {})
            }

            result.append({
                "content": chunk_content.strip(),
                "metadata": chunk_metadata
            })

        logger.info("Семантический чанкинг завершен", chunks_count=len(result))
        return result

    async def _chunk_by_title(
        self,
        content: str,
        metadata: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Чанкинг по заголовкам с использованием регулярных выражений."""

        logger.debug("Используем title-based чанкинг")

        # Ищем заголовки разных уровней
        title_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown заголовки
            r'^.+\n={3,}$',    # Подчеркивание =
            r'^.+\n-{3,}$',    # Подчеркивание -
            r'^\d+\.\s+.+$',   # Нумерованные заголовки
            r'^[А-ЯA-Z][^.!?]*:$',  # Заголовки с двоеточием
        ]

        # Разбиваем по заголовкам
        sections = self._split_by_titles(content, title_patterns)

        # Если разделов мало, используем альтернативную стратегию
        if len(sections) < 2:
            logger.debug("Заголовки не найдены, используем paragraph-based разбивку")
            sections = self._split_by_paragraphs(content)

        # Обрабатываем каждую секцию
        result = []
        for i, (title, section_content) in enumerate(sections):
            # Если секция слишком большая, дополнительно разбиваем
            if len(section_content) > self.chunk_size:
                sub_chunks = await self._split_large_section(section_content)

                for j, sub_chunk in enumerate(sub_chunks):
                    chunk_metadata = {
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(result),
                        "chunking_strategy": "title",
                        "chunk_type": "title_section",
                        "section_title": title,
                        "sub_chunk_index": j,
                        "character_count": len(sub_chunk),
                        "estimated_tokens": self._estimate_tokens(sub_chunk),
                        **(metadata or {})
                    }

                    result.append({
                        "content": sub_chunk.strip(),
                        "metadata": chunk_metadata
                    })
            else:
                chunk_metadata = {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "chunking_strategy": "title",
                    "chunk_type": "title_section",
                    "section_title": title,
                    "character_count": len(section_content),
                    "estimated_tokens": self._estimate_tokens(section_content),
                    **(metadata or {})
                }

                result.append({
                    "content": section_content.strip(),
                    "metadata": chunk_metadata
                })

        logger.info("Title-based чанкинг завершен", chunks_count=len(result))
        return result

    async def _chunk_hybrid(
        self,
        content: str,
        metadata: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Гибридный подход: комбинация структурного и семантического."""

        logger.debug("Используем гибридный чанкинг")

        # Сначала пробуем структурную разбивку
        title_chunks = await self._chunk_by_title(content, metadata)

        # Если получилось мало чанков или они слишком большие, применяем семантическую разбивку
        final_chunks = []

        for chunk in title_chunks:
            chunk_content = chunk["content"]
            chunk_tokens = chunk["metadata"]["estimated_tokens"]

            if chunk_tokens > self.chunk_size * 1.5:  # Слишком большой чанк
                logger.debug("Разбиваем большой чанк семантически",
                           chunk_tokens=chunk_tokens)

                # Применяем семантическую разбивку к большому чанку
                semantic_chunks = await self._chunk_semantic(chunk_content, metadata)

                for i, sem_chunk in enumerate(semantic_chunks):
                    # Обновляем метаданные для гибридного чанка
                    sem_chunk["metadata"]["chunking_strategy"] = "hybrid"
                    sem_chunk["metadata"]["chunk_type"] = "hybrid_semantic"
                    sem_chunk["metadata"]["parent_title"] = chunk["metadata"].get("section_title", "")
                    sem_chunk["metadata"]["hybrid_index"] = i

                    final_chunks.append(sem_chunk)
            else:
                # Чанк подходящего размера
                chunk["metadata"]["chunking_strategy"] = "hybrid"
                chunk["metadata"]["chunk_type"] = "hybrid_title"
                final_chunks.append(chunk)

        # Обновляем индексы
        for i, chunk in enumerate(final_chunks):
            chunk["metadata"]["chunk_index"] = i
            chunk["metadata"]["total_chunks"] = len(final_chunks)

        logger.info("Гибридный чанкинг завершен", chunks_count=len(final_chunks))
        return final_chunks

    async def _apply_semantic_logic(self, sentence_chunks: list[str]) -> list[str]:
        """Применение семантической логики к предложениям."""

        # Простая семантическая логика на основе длины и структуры
        semantic_chunks = []
        current_chunk = ""

        for sentence in sentence_chunks:
            # Проверяем, можно ли добавить предложение к текущему чанку
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Сохраняем текущий чанк и начинаем новый
                if current_chunk:
                    semantic_chunks.append(current_chunk)
                current_chunk = sentence

        # Добавляем последний чанк
        if current_chunk:
            semantic_chunks.append(current_chunk)

        return semantic_chunks

    def _split_by_titles(self, content: str, patterns: list[str]) -> list[tuple[str, str]]:
        """Разбивка текста по заголовкам."""

        lines = content.split('\n')
        sections = []
        current_title = "Введение"
        current_content = ""

        for line in lines:
            is_title = False

            # Проверяем каждый паттерн заголовка
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    is_title = True
                    break

            if is_title and current_content.strip():
                # Сохраняем предыдущую секцию
                sections.append((current_title, current_content.strip()))
                current_title = line.strip()
                current_content = ""
            else:
                current_content += line + "\n"

        # Добавляем последнюю секцию
        if current_content.strip():
            sections.append((current_title, current_content.strip()))

        return sections

    def _split_by_paragraphs(self, content: str) -> list[tuple[str, str]]:
        """Разбивка по параграфам как fallback."""

        paragraphs = re.split(r'\n\s*\n', content)
        sections = []

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                title = f"Параграф {i+1}"
                sections.append((title, paragraph.strip()))

        return sections

    async def _split_large_section(self, content: str) -> list[str]:
        """Разбивка больших секций на меньшие части."""

        # Используем recursive chunker для разбивки больших секций  
        chunks = self.recursive_chunker.chunk(content)

        # Фильтруем слишком маленькие чанки
        result = []
        for chunk in chunks:
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                result.append(chunk_text.strip())

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Приблизительная оценка количества токенов."""
        # Простая эвристика: ~4 символа на токен для русского/английского
        return len(text) // 4

    # Методы для обратной совместимости
    async def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Обратная совместимость с существующим API."""
        return await self.chunk_document(text, metadata)
