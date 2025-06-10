"""Модуль для построения RAG векторной базы."""

import json
import re
import uuid
from pathlib import Path

import faiss
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from .protocols import Document
from .semantic_chunker import SemanticChunker

logger = structlog.get_logger(__name__)


def split_text_with_overlap(text: str, chunk_size: int = 1500, overlap_size: int = 200) -> list[str]:
    """Разбить текст на чанки с перекрытием."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Если это не последний чанк, попробуем найти хорошее место для разделения
        if end < len(text):
            # Ищем ближайший разделитель (точка, новая строка, пробел)
            good_split = -1
            for i in range(end - 100, end):
                if i >= 0 and text[i] in '.!?\n':
                    good_split = i + 1
                    break

            if good_split != -1:
                end = good_split

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Следующий чанк начинается с перекрытием
        start = max(start + chunk_size - overlap_size, end)

        if start >= len(text):
            break

    return chunks


def preprocess_text(text: str) -> str:
    """Предобработка текста."""
    if not text:
        return ""

    # Убираем лишние символы форматирования
    text = text.replace("false", "").replace("true", "")

    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


class RAGBuilder:
    """Построитель RAG векторной базы."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        chunk_size: int = 1500,
        overlap_size: int = 200,
        batch_size: int = 32,
        use_semantic_chunking: bool = True,
        similarity_threshold: float = 0.7
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.batch_size = batch_size
        self.use_semantic_chunking = use_semantic_chunking
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self.semantic_chunker = None

    async def build_from_confluence_data(
        self,
        input_file: Path,
        output_dir: Path
    ) -> None:
        """Построить RAG базу из данных Confluence."""
        logger.info("🚀 Начинаем построение RAG векторной базы")

        # Создаем выходную директорию
        output_dir.mkdir(parents=True, exist_ok=True)

        # Загружаем данные
        logger.info("📖 Загружаем данные из Confluence", file=str(input_file))
        with open(input_file, encoding='utf-8') as f:
            confluence_data = json.load(f)

        logger.info("📊 Данные загружены", total_pages=len(confluence_data))

        # Загружаем модель эмбеддингов на CPU
        logger.info("🧠 Загружаем модель эмбеддингов на CPU", model=self.model_name)
        device = "cpu"  # Принудительно используем CPU
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_model.to(device)
        logger.info("✅ Модель эмбеддингов загружена на CPU")

        # Инициализируем семантический разбивщик с современными техниками
        if self.use_semantic_chunking:
            self.semantic_chunker = SemanticChunker(
                embedding_model=self.embedding_model,
                max_chunk_size=self.chunk_size,
                min_chunk_size=100,
                similarity_threshold=self.similarity_threshold,
                overlap_sentences=2,
                window_size=3,              # Скользящее окно для анализа связности
                adaptive_threshold=True,    # Адаптивный порог на основе статистики
                hierarchical_chunking=True  # Иерархическая разбивка больших документов
            )
            logger.info("✅ Продвинутый семантический разбивщик инициализирован с новыми техниками")

        # Обрабатываем документы
        documents = await self._process_documents(confluence_data)

        # Создаем эмбеддинги
        embeddings = await self._create_embeddings(documents, device)

        # Создаем и сохраняем FAISS индекс
        await self._create_and_save_index(documents, embeddings, output_dir)

        # Тестируем поиск
        await self._test_search(embeddings, documents, device)

    async def _process_documents(self, confluence_data: list[dict]) -> list[Document]:
        """Обработать документы из Confluence."""
        documents = []
        doc_count = 0

        for page in confluence_data:
            content = page.get("page_content", "")
            metadata = page.get("metadata", {})

            # Предобрабатываем текст
            content = preprocess_text(content)

            if not content or len(content) < 50:  # Пропускаем очень короткие документы
                logger.debug("Пропускаем короткий документ", title=metadata.get("title", ""))
                continue

            title = metadata.get("title", "Без названия")
            source = metadata.get("source", "")
            page_id = metadata.get("id", "")

            # Базовые метаданные для всех чанков
            base_metadata = {
                "title": title,
                "source": source,
                "page_id": page_id,
                "when": metadata.get("when", "")
            }

            # Выбираем тип разбивки
            if self.use_semantic_chunking and self.semantic_chunker:
                logger.debug("Используем семантическую разбивку", title=title)
                chunk_dicts = await self.semantic_chunker.chunk_text(content, base_metadata)
                chunks = [chunk_dict["content"] for chunk_dict in chunk_dicts]
                chunks_metadata = [chunk_dict["metadata"] for chunk_dict in chunk_dicts]
            else:
                logger.debug("Используем простую разбивку", title=title)
                chunks = split_text_with_overlap(content, self.chunk_size, self.overlap_size)
                chunks_metadata = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_type": "overlap"
                    })
                    chunks_metadata.append(chunk_metadata)

            logger.debug("Обрабатываем документ",
                        title=title,
                        original_length=len(content),
                        chunks_count=len(chunks),
                        chunking_type="semantic" if self.use_semantic_chunking else "overlap")

            for i, (chunk, chunk_metadata) in enumerate(zip(chunks, chunks_metadata, strict=False)):
                # Создаем уникальный ID для чанка
                chunk_id = f"{page_id}_{i}" if page_id else str(uuid.uuid4())

                # Создаем документ
                doc = Document(
                    id=chunk_id,
                    content=chunk,
                    metadata=chunk_metadata
                )

                documents.append(doc)
                doc_count += 1

        logger.info("📄 Документы обработаны",
                   total_documents=doc_count,
                   from_pages=len([p for p in confluence_data if p.get("page_content")]),
                   chunking_method="semantic" if self.use_semantic_chunking else "overlap")

        return documents

    async def _create_embeddings(self, documents: list[Document], device: str) -> list[np.ndarray]:
        """Создать эмбеддинги для документов."""
        logger.info("🔢 Создаем эмбеддинги для документов")
        texts = [doc.content for doc in documents]

        # Создаем эмбеддинги батчами для экономии памяти
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.debug("Обрабатываем батч", batch=f"{i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")

            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                device=device
            )
            embeddings.extend(batch_embeddings)

        # Сохраняем эмбеддинги в документы
        for doc, embedding in zip(documents, embeddings, strict=False):
            doc.embedding = embedding.tolist()

        logger.info("✅ Эмбеддинги созданы", total_embeddings=len(embeddings))
        return embeddings

    async def _create_and_save_index(
        self,
        documents: list[Document],
        embeddings: list[np.ndarray],
        output_dir: Path
    ) -> None:
        """Создать и сохранить FAISS индекс."""
        logger.info("🗄️ Создаем FAISS индекс")

        # Конвертируем эмбеддинги в numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Нормализуем для cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Создаем индекс
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product для cosine similarity
        index.add(embeddings_array)

        logger.info("✅ FAISS индекс создан", vectors_count=index.ntotal)

        # Сохраняем индекс
        index_file = output_dir / "index.faiss"
        docs_file = output_dir / "documents.json"

        logger.info("💾 Сохраняем индекс и документы")

        # Сохраняем FAISS индекс
        faiss.write_index(index, str(index_file))

        # Сохраняем документы
        docs_data = {}
        for i, doc in enumerate(documents):
            docs_data[str(i)] = {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding
            }

        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        logger.info("🎉 RAG векторная база успешно создана!",
                   index_file=str(index_file),
                   docs_file=str(docs_file),
                   total_vectors=index.ntotal)

    async def _test_search(
        self,
        embeddings: list[np.ndarray],
        documents: list[Document],
        device: str
    ) -> None:
        """Тестировать поиск в созданной базе."""
        logger.info("🔍 Тестируем поиск...")

        # Создаем простой индекс для тестирования
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        test_index = faiss.IndexFlatIP(self.embedding_dim)
        test_index.add(embeddings_array)

        test_query = "Jenkins URL"
        test_embedding = self.embedding_model.encode([test_query], convert_to_tensor=False, device=device)
        test_embedding = np.array(test_embedding, dtype=np.float32)
        faiss.normalize_L2(test_embedding)

        scores, indices = test_index.search(test_embedding, 5)

        logger.info("🎯 Результаты тестового поиска", query=test_query)
        for i, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False)):
            if idx != -1:
                doc = documents[idx]
                logger.info(f"Результат {i+1}",
                           score=float(score),
                           title=doc.metadata.get("title", ""),
                           content_preview=doc.content[:200] + "...")


async def build_rag_from_confluence(
    input_file: str = "data/confluence_data.json",
    output_dir: str = "data/faiss_index"
) -> None:
    """Удобная функция для построения RAG базы."""
    builder = RAGBuilder()
    await builder.build_from_confluence_data(
        Path(input_file),
        Path(output_dir)
    )
