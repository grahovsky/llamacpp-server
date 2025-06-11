"""Современный сервис эмбеддингов с поддержкой различных провайдеров."""

import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, Protocol

import structlog
import torch
from sentence_transformers import SentenceTransformer

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from model2vec import StaticModel
    HAS_MODEL2VEC = True
except ImportError:
    HAS_MODEL2VEC = False

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingProvider(Enum):
    """Провайдеры эмбеддингов."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai" 
    MODEL2VEC = "model2vec"


class EmbeddingProtocol(Protocol):
    """Протокол для провайдеров эмбеддингов."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов."""
        ...

    @abstractmethod
    async def is_ready(self) -> bool:
        """Проверить готовность провайдера."""
        ...


class SentenceTransformerProvider:
    """Провайдер эмбеддингов через SentenceTransformers."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self._model: SentenceTransformer | None = None

    def _get_device(self, device: str) -> str:
        """Определение устройства для модели."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    async def _ensure_model_loaded(self) -> None:
        """Ленивая загрузка модели."""
        if self._model is None:
            logger.info("Загрузка SentenceTransformer модели",
                       model=self.model_name, device=self.device)

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True
                )
            )

            logger.info("SentenceTransformer модель загружена")

    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста."""
        await self._ensure_model_loaded()

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, show_progress_bar=False)
        )

        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов."""
        await self._ensure_model_loaded()

        logger.debug("Создание batch эмбеддингов", count=len(texts))

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, show_progress_bar=False)
        )

        return [emb.tolist() for emb in embeddings]

    async def is_ready(self) -> bool:
        """Проверить готовность провайдера."""
        try:
            await self._ensure_model_loaded()
            return self._model is not None
        except Exception as e:
            logger.error("Ошибка проверки готовности SentenceTransformer", error=str(e))
            return False


class OpenAIProvider:
    """Провайдер эмбеддингов через OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        if not HAS_OPENAI:
            raise ImportError("openai package не установлен. Установите: pip install openai")

        self.model_name = model_name
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста через OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error("Ошибка получения эмбеддинга от OpenAI", error=str(e))
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов через OpenAI API."""
        logger.debug("Создание batch эмбеддингов через OpenAI", count=len(texts))

        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error("Ошибка получения batch эмбеддингов от OpenAI", error=str(e))
            raise

    async def is_ready(self) -> bool:
        """Проверить готовность провайдера."""
        try:
            # Простая проверка с коротким текстом
            await self.embed_text("test")
            return True
        except Exception:
            return False


class Model2VecProvider:
    """Провайдер эмбеддингов через Model2Vec (сверхбыстрые статические эмбеддинги)."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        if not HAS_MODEL2VEC:
            raise ImportError("model2vec package не установлен. Установите: pip install model2vec")

        self.model_name = model_name
        self._model: StaticModel | None = None

    async def _ensure_model_loaded(self) -> None:
        """Ленивая загрузка модели."""
        if self._model is None:
            logger.info("Загрузка Model2Vec модели", model=self.model_name)

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: StaticModel.from_pretrained(self.model_name)
            )

            logger.info("Model2Vec модель загружена")

    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста."""
        await self._ensure_model_loaded()

        embedding = self._model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов."""
        await self._ensure_model_loaded()

        logger.debug("Создание batch эмбеддингов через Model2Vec", count=len(texts))

        embeddings = self._model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    async def is_ready(self) -> bool:
        """Проверить готовность провайдера."""
        try:
            await self._ensure_model_loaded()
            return self._model is not None
        except Exception as e:
            logger.error("Ошибка проверки готовности Model2Vec", error=str(e))
            return False


class ModernEmbeddingService:
    """Современный сервис эмбеддингов с поддержкой различных провайдеров."""

    def __init__(
        self,
        provider_type: str = "sentence_transformers",  # sentence_transformers, openai, model2vec
        model_name: str | None = None,
        settings=None,
        **kwargs
    ):
        self.provider_type = provider_type
        self.settings = settings or get_settings()

        # Определяем модель по умолчанию
        if model_name is None:
            model_name = self._get_default_model(provider_type)

        self.provider = self._create_provider(provider_type, model_name, **kwargs)

    def _get_default_model(self, provider_type: str) -> str:
        """Получить модель по умолчанию для провайдера."""
        defaults = {
            "sentence_transformers": "BAAI/bge-m3",  # Лучшая multilingual модель 2025
            "openai": "text-embedding-3-small",
            "model2vec": "minishlab/potion-base-8M"
        }
        return defaults.get(provider_type, "BAAI/bge-m3")

    def _create_provider(
        self,
        provider_type: str,
        model_name: str,
        **kwargs
    ) -> EmbeddingProtocol:
        """Создать провайдер эмбеддингов."""

        if provider_type == "sentence_transformers":
            device = kwargs.get("device", self.settings.get_embedding_device())
            return SentenceTransformerProvider(model_name, device)

        elif provider_type == "openai":
            api_key = kwargs.get("api_key", getattr(self.settings, "openai_api_key", None))
            return OpenAIProvider(model_name, api_key)

        elif provider_type == "model2vec":
            return Model2VecProvider(model_name)

        else:
            raise ValueError(f"Неизвестный провайдер: {provider_type}")

    async def embed_text(self, text: str) -> list[float]:
        """Получить эмбеддинг для текста."""
        return await self.provider.embed_text(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для пакета текстов."""
        return await self.provider.embed_batch(texts)

    async def is_ready(self) -> bool:
        """Проверить готовность сервиса."""
        return await self.provider.is_ready()

    async def initialize(self) -> None:
        """Инициализация сервиса эмбеддингов."""
        # Проверяем готовность провайдера
        if not await self.is_ready():
            raise RuntimeError(f"Провайдер {self.provider_type} не готов к работе")
        
        logger.info("✅ Сервис эмбеддингов инициализирован", 
                   provider=self.provider_type,
                   model=getattr(self.provider, "model_name", "unknown"))

    async def get_embedding_dimension(self) -> int:
        """Получить размерность эмбеддингов."""
        # Тестовый эмбеддинг для определения размерности
        test_embedding = await self.embed_text("test")
        return len(test_embedding)

    def get_provider_info(self) -> dict[str, Any]:
        """Получить информацию о провайдере."""
        return {
            "provider_type": self.provider_type,
            "model_name": getattr(self.provider, "model_name", "unknown"),
            "device": getattr(self.provider, "device", "unknown")
        }


# Фабричная функция для удобства
def create_embedding_service(
    provider: str = "sentence_transformers",
    model: str | None = None,
    **kwargs
) -> ModernEmbeddingService:
    """Создать сервис эмбеддингов с указанным провайдером."""
    return ModernEmbeddingService(
        provider_type=provider,
        model_name=model,
        **kwargs
    )
