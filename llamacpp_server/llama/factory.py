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


def _get_cuda_device_count() -> int:
    """Получить количество доступных CUDA устройств."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            return len([line for line in result.stdout.strip().split('\n') if line.startswith('GPU')])
    except FileNotFoundError:
        pass
    return 0


class MockLlama:
    """Mock LLama для разработки."""
    
    def tokenize(self, text: bytes) -> list[int]:
        """Mock токенизация для подсчета токенов."""
        # Простая оценка: 1 токен ≈ 4 символа для латиницы, 2 символа для кириллицы
        text_str = text.decode('utf-8', errors='ignore')
        
        # Подсчитываем символы
        char_count = len(text_str)
        
        # Эмулируем токены (примерно 1 токен на 3-4 символа)
        estimated_tokens = max(1, char_count // 3)
        
        # Возвращаем список фиктивных токенов
        return list(range(estimated_tokens))
    
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
        
        # Диагностика GPU поддержки
        try:
            from llama_cpp import llama_cpp
            gpu_support = hasattr(llama_cpp, 'llama_supports_gpu_offload') and llama_cpp.llama_supports_gpu_offload()
            logger.info("GPU диагностика", 
                       gpu_support=gpu_support, 
                       n_gpu_layers=settings.n_gpu_layers,
                       cuda_device_count=_get_cuda_device_count())
        except Exception as e:
            logger.warning("Ошибка GPU диагностики", error=str(e))
        
        try:
            # Подготавливаем параметры
            llama_params = {
                "model_path": str(settings.model_path),
                "n_ctx": settings.n_ctx,
                "n_batch": settings.n_batch,
                "n_threads": settings.n_threads,
                "n_gpu_layers": settings.n_gpu_layers,
                "use_mmap": settings.use_mmap,
                "use_mlock": settings.use_mlock,
                "chat_format": settings.chat_format,
                "verbose": settings.verbose,
            }
            
            # Добавляем GPU параметры если есть GPU поддержка
            if settings.n_gpu_layers > 0:
                llama_params["main_gpu"] = settings.main_gpu
                
                # Обработка tensor_split
                if settings.tensor_split:
                    try:
                        tensor_split = [float(x.strip()) for x in settings.tensor_split.split(',')]
                        llama_params["tensor_split"] = tensor_split
                        logger.info("Настройка tensor_split", tensor_split=tensor_split)
                    except ValueError as e:
                        logger.warning("Неверный формат tensor_split", error=str(e))
            
            llama = Llama(**llama_params)
            
            logger.info("LLama модель успешно инициализирована", 
                       n_gpu_layers=settings.n_gpu_layers)
            return llama
            
        except Exception as e:
            logger.error("Ошибка инициализации LLama модели", error=str(e))
            raise 