"""FAISS векторное хранилище."""

import asyncio
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import structlog

from ..config.settings import get_settings
from .protocols import Document, SearchResult

logger = structlog.get_logger(__name__)


class FaissVectorStore:
    """FAISS векторное хранилище с async интерфейсом."""

    def __init__(self) -> None:
        self._index: faiss.Index = None
        self._documents: dict[int, Document] = {}
        self._dimension: int = 1024  # BGE-M3 dimension
        settings = get_settings()
        self._index_dir = Path(settings.faiss_index_path)

    async def _ensure_index_loaded(self) -> None:
        """Загружаем индекс при первом использовании."""
        if self._index is None:
            logger.info("Проверка FAISS индекса", index_dir=str(self._index_dir))

            # Проверяем разные варианты названий файлов
            possible_files = [
                # Наш новый формат (приоритет)
                (self._index_dir / "index.faiss", self._index_dir / "documents.json"),
            ]

            index_loaded = False
            for index_file, docs_file in possible_files:
                logger.debug("Проверка файлов", index_file=str(index_file), docs_file=str(docs_file))

                if index_file.exists():
                    logger.info("Найден FAISS индекс",
                               index_file=str(index_file),
                               docs_file=str(docs_file),
                               docs_exists=docs_file.exists())

                    try:
                        await self._load_index_files(index_file, docs_file)
                        index_loaded = True
                        break
                    except Exception as e:
                        logger.warning("Ошибка загрузки индекса",
                                     index_file=str(index_file),
                                     error=str(e))
                        continue

            if not index_loaded:
                # Создаем новый индекс
                logger.info("Создание нового FAISS индекса", dimension=self._dimension)
                loop = asyncio.get_event_loop()
                self._index = await loop.run_in_executor(
                    None,
                    lambda: faiss.IndexFlatIP(self._dimension)  # Inner product для cosine similarity
                )

    async def add_documents(self, documents: list[Document], batch_size: int = 100) -> None:
        """Добавить документы в хранилище с батчевой обработкой."""
        await self._ensure_index_loaded()

        logger.info("🔄 Добавление документов в FAISS с батчевой обработкой", 
                   count=len(documents), 
                   batch_size=batch_size)

        # Обрабатываем документы батчами для оптимизации памяти
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_end = min(i + batch_size, len(documents))
            
            logger.debug("Обработка батча", 
                        batch=f"{i + 1}-{batch_end}",
                        total=len(documents))

            # Подготавливаем эмбеддинги для батча
            embeddings = []
            for doc in batch_docs:
                if doc.embedding is None:
                    raise ValueError(f"Document {doc.id} missing embedding")
                embeddings.append(doc.embedding)

            # Нормализуем для cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)

            # Добавляем батч в индекс
            loop = asyncio.get_event_loop()
            current_size = self._index.ntotal

            await loop.run_in_executor(
                None,
                self._index.add,
                embeddings_array
            )

            # Сохраняем документы батча
            for j, doc in enumerate(batch_docs):
                self._documents[current_size + j] = doc

            total_added += len(batch_docs)
            logger.debug("✅ Батч обработан",
                        batch_docs=len(batch_docs),
                        total_added=total_added,
                        total_in_index=self._index.ntotal)

        logger.info("🎉 Все документы добавлены", 
                   total_docs=self._index.ntotal,
                   batches_processed=len(range(0, len(documents), batch_size)))

    async def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        """Поиск похожих документов."""
        await self._ensure_index_loaded()

        if self._index.ntotal == 0:
            logger.warning("Поиск в пустом индексе")
            return []

        logger.debug("Поиск в FAISS", k=k, total_docs=self._index.ntotal)

        # Нормализуем запрос для cosine similarity
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Выполняем поиск в thread pool
        loop = asyncio.get_event_loop()
        scores, indices = await loop.run_in_executor(
            None,
            self._index.search,
            query_array,
            k
        )

        # Детальное логирование результатов поиска
        logger.debug("🔍 FAISS Search Results",
                    raw_scores=scores[0].tolist() if len(scores) > 0 else [],
                    raw_indices=indices[0].tolist() if len(indices) > 0 else [],
                    available_docs=list(self._documents.keys())[:10])  # Первые 10 ключей для диагностики

        # Формируем результаты
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:  # FAISS возвращает -1 для отсутствующих результатов
                logger.debug("Пропуск результата с индексом -1")
                continue
            if idx in self._documents:
                doc = self._documents[idx]
                result = SearchResult(document=doc, score=float(score))
                results.append(result)

                logger.debug("Найден документ",
                           doc_id=doc.id,
                           score=float(score),
                           content_preview=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
            else:
                logger.warning("Индекс не найден в документах", idx=int(idx), available_keys=list(self._documents.keys())[:5])

        logger.debug("Поиск завершен", found=len(results))
        return results

    async def _load_index_files(self, index_file: Path, docs_file: Path) -> None:
        """Загрузить индекс из конкретных файлов."""
        logger.info("Загрузка FAISS индекса",
                   index_file=str(index_file),
                   docs_file=str(docs_file))

        # Загружаем индекс в thread pool
        loop = asyncio.get_event_loop()
        self._index = await loop.run_in_executor(
            None,
            faiss.read_index,
            str(index_file)
        )

        # Загружаем документы в зависимости от формата
        if docs_file.exists():
            if docs_file.suffix == '.json':
                # Наш новый JSON формат
                with open(docs_file, encoding='utf-8') as f:
                    docs_data = json.load(f)

                    # Проверяем формат данных
                    logger.debug("Проверяем формат документов",
                                sample_keys=list(docs_data.keys())[:5],
                                total_keys=len(docs_data))

                    if isinstance(docs_data, dict) and all(isinstance(v, dict) for v in docs_data.values()):
                        # Это наш новый формат {index: {id, content, metadata, embedding}}
                        self._documents = {}
                        for i, doc_data in docs_data.items():
                            self._documents[int(i)] = Document(
                                id=doc_data["id"],
                                content=doc_data["content"],
                                metadata=doc_data["metadata"],
                                embedding=doc_data.get("embedding")
                            )
                        logger.info("Загружены документы из нового JSON формата", count=len(self._documents))
                    else:
                        # Старый формат
                        self._documents = {
                            int(k): Document(**v) for k, v in docs_data.items()
                        }
                        logger.info("Загружены документы из старого JSON формата", count=len(self._documents))
            elif docs_file.suffix == '.pkl':
                # Pickle формат (стандартный для многих библиотек)
                with open(docs_file, 'rb') as f:
                    docs_data = pickle.load(f)

                    logger.info("Анализ pickle данных",
                               type=type(docs_data).__name__,
                               has_keys=hasattr(docs_data, 'keys'),
                               length=len(docs_data) if hasattr(docs_data, '__len__') else 'N/A')

                    # Если это LangChain объект, попробуем извлечь документы
                    if hasattr(docs_data, 'docstore') and hasattr(docs_data.docstore, '_dict'):
                        logger.info("Обнаружен LangChain docstore")
                        langchain_docs = docs_data.docstore._dict
                        self._documents = {}
                        for i, (doc_id, doc) in enumerate(langchain_docs.items()):
                            if hasattr(doc, 'page_content'):
                                self._documents[i] = Document(
                                    id=str(doc_id),
                                    content=doc.page_content,
                                    metadata=getattr(doc, 'metadata', {}),
                                    embedding=None
                                )
                            else:
                                self._documents[i] = Document(
                                    id=str(doc_id),
                                    content=str(doc),
                                    metadata={},
                                    embedding=None
                                )
                        logger.info("Загружены документы из LangChain", count=len(self._documents))

                    # Обрабатываем разные форматы pickle данных
                    elif isinstance(docs_data, dict):
                        # Если это словарь документов
                        self._documents = {}
                        for i, (doc_id, content) in enumerate(docs_data.items()):
                            if isinstance(content, dict):
                                # Если content уже структурированный
                                self._documents[i] = Document(
                                    id=str(doc_id),
                                    content=content.get('content', str(content)),
                                    metadata=content.get('metadata', {}),
                                    embedding=content.get('embedding')
                                )
                            else:
                                # Если content это просто текст
                                self._documents[i] = Document(
                                    id=str(doc_id),
                                    content=str(content),
                                    metadata={},
                                    embedding=None
                                )
                    elif isinstance(docs_data, list):
                        # Если это список документов
                        self._documents = {}
                        for i, doc in enumerate(docs_data):
                            if isinstance(doc, dict):
                                self._documents[i] = Document(
                                    id=doc.get('id', str(i)),
                                    content=doc.get('content', str(doc)),
                                    metadata=doc.get('metadata', {}),
                                    embedding=doc.get('embedding')
                                )
                            else:
                                self._documents[i] = Document(
                                    id=str(i),
                                    content=str(doc),
                                    metadata={},
                                    embedding=None
                                )
                    elif isinstance(docs_data, tuple):
                        # Tuple формат - часто это (index, docstore) или (docs, embeddings)
                        logger.info("Обнаружен tuple формат", items=len(docs_data))

                        # Попробуем разные варианты
                        for i, item in enumerate(docs_data):
                            logger.info(f"Tuple элемент {i}",
                                       type=type(item).__name__,
                                       has_len=hasattr(item, '__len__'),
                                       length=len(item) if hasattr(item, '__len__') else 'N/A')

                        # Если это LangChain FAISS формат (vectorstore, docstore)
                        if len(docs_data) >= 2:
                            # Обычно второй элемент - это docstore
                            possible_docstore = docs_data[1] if len(docs_data) > 1 else docs_data[0]

                            if hasattr(possible_docstore, '_dict'):
                                logger.info("Найден docstore в tuple")
                                langchain_docs = possible_docstore._dict
                                self._documents = {}
                                for i, (doc_id, doc) in enumerate(langchain_docs.items()):
                                    if hasattr(doc, 'page_content'):
                                        self._documents[i] = Document(
                                            id=str(doc_id),
                                            content=doc.page_content,
                                            metadata=getattr(doc, 'metadata', {}),
                                            embedding=None
                                        )
                                    else:
                                        self._documents[i] = Document(
                                            id=str(doc_id),
                                            content=str(doc),
                                            metadata={},
                                            embedding=None
                                        )
                                logger.info("Загружены документы из tuple docstore", count=len(self._documents))
                            elif isinstance(possible_docstore, (list, dict)):
                                # Прямой список/словарь документов
                                self._documents = {}
                                if isinstance(possible_docstore, dict):
                                    docs_iter = possible_docstore.items()
                                else:
                                    docs_iter = enumerate(possible_docstore)

                                for i, (doc_id, doc) in enumerate(docs_iter):
                                    if hasattr(doc, 'page_content'):
                                        content = doc.page_content
                                        metadata = getattr(doc, 'metadata', {})
                                    elif isinstance(doc, dict):
                                        content = doc.get('content', doc.get('text', str(doc)))
                                        metadata = doc.get('metadata', {})
                                    else:
                                        content = str(doc)
                                        metadata = {}

                                    self._documents[i] = Document(
                                        id=str(doc_id),
                                        content=content,
                                        metadata=metadata,
                                        embedding=None
                                    )
                                logger.info("Загружены документы из tuple list/dict", count=len(self._documents))
                            else:
                                logger.warning("Не удалось извлечь документы из tuple")
                                self._documents = {}
                        else:
                            logger.warning("Tuple слишком короткий")
                            self._documents = {}
                    else:
                        logger.warning("Неизвестный формат pickle данных", type=type(docs_data))
                        self._documents = {}
        else:
            logger.warning("Файл документов не найден, создаю пустой словарь")
            self._documents = {}

        logger.info("FAISS индекс загружен",
                   docs_count=len(self._documents),
                   index_size=self._index.ntotal)

    async def load_index(self, index_path: str) -> None:
        """Загрузить существующий индекс."""
        index_dir = Path(index_path)
        index_file = index_dir / "index.faiss"  # Исправлено имя файла
        docs_file = index_dir / "documents.json"

        if not index_file.exists() or not docs_file.exists():
            raise FileNotFoundError(f"Index files not found in {index_path}")

        logger.info("Загрузка FAISS индекса", path=index_path)

        # Загружаем индекс в thread pool
        loop = asyncio.get_event_loop()
        self._index = await loop.run_in_executor(
            None,
            faiss.read_index,
            str(index_file)
        )

        # Загружаем документы в нашем формате
        with open(docs_file, encoding='utf-8') as f:
            docs_data = json.load(f)
            self._documents = {}
            for i, doc_data in docs_data.items():
                self._documents[int(i)] = Document(
                    id=doc_data["id"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    embedding=doc_data.get("embedding")
                )

        logger.info("FAISS индекс загружен",
                   docs_count=len(self._documents),
                   index_size=self._index.ntotal)

    async def save_index(self, index_path: str) -> None:
        """Сохранить индекс."""
        if self._index is None:
            logger.warning("Попытка сохранить пустой индекс")
            return

        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        index_file = index_dir / "faiss.index"
        docs_file = index_dir / "documents.json"

        logger.info("Сохранение FAISS индекса", path=index_path)

        # Сохраняем индекс в thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            faiss.write_index,
            self._index,
            str(index_file)
        )

        # Сохраняем документы
        docs_data = {
            str(k): {
                "id": v.id,
                "content": v.content,
                "metadata": v.metadata,
                "embedding": v.embedding
            }
            for k, v in self._documents.items()
        }

        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        logger.info("FAISS индекс сохранен")
