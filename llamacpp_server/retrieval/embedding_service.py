"""Сервис для работы с эмбеддингами BGE-M3."""

import asyncio

import structlog
import torch
from sentence_transformers import SentenceTransformer

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Сервис для создания эмбеддингов с BGE-M3."""

    def __init__(self) -> None:
        self._model: SentenceTransformer = None
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._device = settings.get_embedding_device()

    async def _ensure_model_loaded(self) -> None:
        """Загружаем модель при первом использовании."""
        if self._model is None:
            logger.info("Загрузка embedding модели", model=self._model_name, device=self._device)
            # Загружаем в thread pool чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self._model_name,
                    trust_remote_code=True,
                    device=self._device
                )
            )

            # Перемещаем на нужное устройство
            if self._device != 'cpu' and torch.cuda.is_available():
                logger.info("Перемещаем embedding модель на GPU", device=self._device)
                await loop.run_in_executor(None, self._model.to, self._device)
            else:
                logger.info("Используем CPU для embedding модели")
                await loop.run_in_executor(None, self._model.to, 'cpu')

            logger.info("Embedding модель загружена", device=self._device)

    async def is_ready(self) -> bool:
        """Проверить готовность модели."""
        try:
            await self._ensure_model_loaded()
            return self._model is not None
        except Exception as e:
            logger.error("Ошибка проверки готовности embedding модели", error=str(e))
            return False

    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста."""
        await self._ensure_model_loaded()

        logger.debug("Создание эмбеддинга", text_length=len(text), device=self._device)

        # Выполняем embedding в thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            self._model.encode,
            text
        )

        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов."""
        await self._ensure_model_loaded()

        logger.debug("Создание пакета эмбеддингов", count=len(texts), device=self._device)

        # Выполняем batch embedding в thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._model.encode,
            texts
        )

        return [emb.tolist() for emb in embeddings]
