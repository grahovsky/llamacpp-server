"""Factory для создания LLama инстансов."""

import structlog
from typing import Optional
from unittest.mock import Mock

try:
    from llama_cpp import Llama
except ImportError:
    logger = structlog.get_logger(__name__)
    logger.warning("llama-cpp-python не установлен, используется mock")
    Llama = None

from ..config import Settings


logger = structlog.get_logger(__name__)


class MockLlama:
    """Mock LLama для разработки."""
    
    def create_completion(self, **kwargs):
        """Mock генерация."""
        if kwargs.get("stream", False):
            import time
            # Генератор с задержками для реалистичного стриминга
            chunks = [
                {"choices": [{"delta": {"content": "Это "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "тестовый "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "ответ "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "от "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "mock "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "модели!"}, "finish_reason": "stop"}]},
            ]
            
            # Возвращаем список чанков, задержка будет в async функции
            return chunks
        else:
            return {
                "choices": [
                    {
                        "text": "Это тестовый ответ от mock LLama модели!",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                },
            }
    
    def create_chat_completion(self, **kwargs):
        """Mock chat completion."""
        if kwargs.get("stream", False):
            return [
                {
                    "choices": [
                        {
                            "delta": {"content": "Привет! "},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {"content": "Как дела?"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ]
        else:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Привет! Это тестовый ответ от mock модели. Как дела?"
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 12,
                    "total_tokens": 17,
                },
            }


class LlamaFactory:
    """Factory для создания LLama инстансов."""
    
    @staticmethod
    async def create_llama(settings: Settings):
        """Создать инстанс LLama модели."""
        if settings.dev_mode:
            logger.info("Инициализация mock LLama модели (dev режим)")
            return MockLlama()
        
        if Llama is None:
            raise ImportError("llama-cpp-python не установлен")
        
        logger.info("Инициализация LLama модели", model_path=str(settings.model_path))
        
        try:
            llama = Llama(
                model_path=str(settings.model_path),
                n_ctx=settings.n_ctx,
                n_batch=settings.n_batch,
                n_threads=settings.n_threads,
                n_gpu_layers=settings.n_gpu_layers,
                use_mmap=settings.use_mmap,
                use_mlock=settings.use_mlock,
                chat_format=settings.chat_format,
                verbose=settings.verbose,
            )
            
            logger.info("LLama модель успешно инициализирована")
            return llama
            
        except Exception as e:
            logger.error("Ошибка инициализации LLama модели", error=str(e))
            raise 