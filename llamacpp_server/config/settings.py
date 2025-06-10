"""Настройки приложения используя Pydantic Settings v2."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Dict, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class RAGDefaults:
    """Дефолтные параметры для RAG-only системы."""
    
    # === RAG параметры (единственные в системе) ===
    TEMPERATURE = 0.5 
    TOP_K = 40
    TOP_P = 0.9
    REPEAT_PENALTY = 1.1      
    MAX_TOKENS = 2048          
    SEED = -1
    
    # === Только для заголовков ===
    TITLE_TEMPERATURE = 0.2
    TITLE_REPEAT_PENALTY = 1.1  
    TITLE_MAX_TOKENS = 50  


class Settings(BaseSettings):
    """Настройки приложения."""

    # === Модель ===
    model_path: Annotated[Path, Field(description="Путь к файлу модели GGUF")] = Path(
        #"models/llama-3.1-8b-q4_k_m.gguf",
        "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        #"models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        # "models/llama2:7b.gguf"
    )

    # === Параметры модели ===
    n_ctx: Annotated[int, Field(ge=1, description="Размер контекста модели")] = 8192
    n_batch: Annotated[int, Field(ge=1, description="Размер батча")] = 512
    n_threads: Annotated[int, Field(ge=1, description="Количество потоков")] = 8
    n_gpu_layers: Annotated[int, Field(ge=0, description="Количество слоев на GPU")] = 30
    main_gpu: Annotated[int, Field(ge=0, description="Основная GPU для вычислений")] = 0
    tensor_split: Annotated[str | None, Field(description="Разделение тензоров между GPU (через запятую)")] = None

    # === Дополнительные параметры производительности ===
    n_ubatch: Annotated[int, Field(ge=1, description="Размер микро-батча для CUDA")] = 512
    flash_attn: Annotated[bool, Field(description="Использовать Flash Attention (если поддерживается)")] = False
    rope_freq_base: Annotated[float, Field(ge=1.0, description="RoPE frequency base")] = 10000.0
    rope_freq_scale: Annotated[float, Field(gt=0.0, description="RoPE frequency scaling factor")] = 1.0

    # === Управление контекстом и токенами ===
    # Формула: n_ctx >= rag_max_context + max_history_tokens + max_response_tokens + safety_buffer
    max_response_tokens: Annotated[int, Field(ge=50, le=2048, description="Максимум токенов в ответе")] = RAGDefaults.MAX_TOKENS
    max_history_tokens: Annotated[int, Field(ge=0, description="Максимум токенов в истории чата")] = 0
    rag_max_context: Annotated[int, Field(ge=0, description="Максимум токенов для RAG контекста")] = 2024
    safety_buffer_tokens: Annotated[int, Field(ge=50, description="Буфер безопасности для непредвиденных токенов")] = 300

    # === Параметры генерации (RAG дефолты) ===
    top_k: Annotated[int, Field(ge=1, description="Top-K sampling")] = RAGDefaults.TOP_K
    top_p: Annotated[float, Field(ge=0.0, le=1.0, description="Top-P (nucleus) sampling")] = RAGDefaults.TOP_P
    repeat_penalty: Annotated[float, Field(ge=0.0, description="Штраф за повторы")] = RAGDefaults.REPEAT_PENALTY
    temperature: Annotated[float, Field(ge=0.0, le=2.0, description="Температура")] = RAGDefaults.TEMPERATURE

    # === Сервер ===
    host: Annotated[str, Field(description="Хост сервера")] = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535, description="Порт сервера")] = 8090

    # === Дополнительные настройки ===
    verbose: Annotated[bool, Field(description="Подробный вывод")] = False
    use_mmap: Annotated[bool, Field(description="Использовать memory mapping")] = True
    use_mlock: Annotated[bool, Field(description="Использовать memory locking")] = False
    chat_format: Annotated[str | None, Field(description="Формат чата (None = автоопределение)")] = None

    # === Логирование ===
    log_level: Annotated[str, Field(description="Уровень логирования")] = "INFO"

    # === Режим разработки ===
    dev_mode: Annotated[bool, Field(description="Режим разработки (без модели)")] = False

    # === RAG настройки (обязательные для RAG-only системы) ===
    embedding_model: str = Field(
        "BAAI/bge-m3",
        description="Модель для создания эмбеддингов"
    )
    faiss_index_path: str = Field(
        "./data/faiss_index",
        description="Путь к FAISS индексу"
    )

    # RAG всегда включен в RAG-only системе
    enable_rag: bool = Field(
        True,
        description="RAG всегда включен (RAG-only система)",
        frozen=True  # Нельзя изменить
    )
    rag_search_k: int = Field(
        15,
        description="Количество документов для поиска в RAG"
    )

    # === Глобальная настройка устройства вычислений ===
    compute_device: Annotated[str, Field(description="Устройство для вычислений (cpu, cuda, auto)")] = "cuda"
    force_cpu_embedding: Annotated[bool, Field(description="Принудительно использовать CPU для embedding моделей")] = True

    # === Тип модели и шаблоны промптов ===
    model_type: Annotated[str, Field(description="Тип модели: instruct, chat")] = "instruct"

    model_config = {
        "env_prefix": "LLAMACPP_",
        "env_file": ".env",  # Читается из корня проекта (рабочая директория)
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    # === RAG Request Builder Methods ===
    def build_rag_params(
        self, 
        is_title: bool = False,
        **overrides: Any
    ) -> Dict[str, Any]:
        """
        Создает параметры для RAG запроса (единственный тип в системе).
        
        Args:
            is_title: True для генерации заголовков
            **overrides: Пользовательские переопределения параметров
            
        Returns:
            Словарь параметров для LLM
        """
        if is_title:
            # Специальные параметры для заголовков
            params = {
                "temperature": RAGDefaults.TITLE_TEMPERATURE,
                "repeat_penalty": RAGDefaults.TITLE_REPEAT_PENALTY, 
                "max_tokens": RAGDefaults.TITLE_MAX_TOKENS,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "seed": RAGDefaults.SEED,
            }
        else:
            # Стандартные RAG параметры
            params = {
                "temperature": self.temperature,
                "repeat_penalty": self.repeat_penalty,
                "max_tokens": RAGDefaults.MAX_TOKENS,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "seed": RAGDefaults.SEED,
            }
        
        # Применяем пользовательские переопределения
        params.update(overrides)
        
        return params

    def merge_request_params(self, user_request, is_title: bool = False) -> Dict[str, Any]:
        """
        Объединяет пользовательские параметры с RAG дефолтами.
        
        Args:
            user_request: Объект запроса пользователя с параметрами
            is_title: True для генерации заголовков
            
        Returns:
            Финальные параметры для LLM
        """
        # Получаем базовые RAG параметры
        base_params = self.build_rag_params(is_title)
        
        # Создаем словарь пользовательских параметров (только не-None)
        user_overrides = {}
        
        if hasattr(user_request, 'temperature') and user_request.temperature is not None:
            user_overrides['temperature'] = user_request.temperature
        if hasattr(user_request, 'top_k') and user_request.top_k is not None:
            user_overrides['top_k'] = user_request.top_k
        if hasattr(user_request, 'top_p') and user_request.top_p is not None:
            user_overrides['top_p'] = user_request.top_p
        if hasattr(user_request, 'repeat_penalty') and user_request.repeat_penalty is not None:
            user_overrides['repeat_penalty'] = user_request.repeat_penalty
        if hasattr(user_request, 'max_tokens') and user_request.max_tokens is not None:
            user_overrides['max_tokens'] = user_request.max_tokens
        if hasattr(user_request, 'seed') and user_request.seed is not None:
            user_overrides['seed'] = user_request.seed
            
        # Объединяем базовые параметры с пользовательскими
        base_params.update(user_overrides)
        
        return base_params

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        """Проверка существования файла модели."""
        # Временно отключаем проверку для отладки
        # if not v.exists():
        #     raise ValueError(f"Файл модели не найден: {v}")
        return v

    @field_validator("max_history_tokens")
    @classmethod
    def validate_history_tokens(cls, v: int, info) -> int:
        """Валидация размера истории относительно общего контекста."""
        if hasattr(info.data, 'n_ctx') and info.data.get('n_ctx'):
            n_ctx = info.data['n_ctx']
            if v > n_ctx * 0.7:  # История не должна занимать больше 70% контекста
                raise ValueError(f"max_history_tokens ({v}) слишком большой для n_ctx ({n_ctx})")
        return v

    @field_validator("compute_device")
    @classmethod
    def validate_compute_device(cls, v: str) -> str:
        """Валидация устройства вычислений."""
        valid_devices = {"cpu", "cuda", "auto"}
        if v.lower() not in valid_devices:
            raise ValueError(f"compute_device должен быть одним из: {valid_devices}")
        return v.lower()

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Валидация типа модели."""
        valid_types = {"instruct", "chat", "auto"}
        if v.lower() not in valid_types:
            raise ValueError(f"model_type должен быть одним из: {valid_types}")
        return v.lower()

    def validate_context_allocation(self) -> None:
        """Проверка корректности распределения токенов контекста."""
        total_allocated = (
            self.max_history_tokens +
            self.max_response_tokens +
            self.rag_max_context +
            self.safety_buffer_tokens
        )

        if total_allocated > self.n_ctx:
            raise ValueError(
                f"Сумма токенов превышает размер контекста: "
                f"{total_allocated} > {self.n_ctx}. "
                f"История: {self.max_history_tokens}, "
                f"Ответ: {self.max_response_tokens}, "
                f"RAG: {self.rag_max_context}, "
                f"Буфер: {self.safety_buffer_tokens}"
            )

    @model_validator(mode="after")
    def validate_context_allocation_after(self):
        """Проверка корректности распределения токенов контекста после инициализации."""
        self.validate_context_allocation()
        return self

    @property
    def available_context_tokens(self) -> int:
        """Доступные токены контекста после резервирования."""
        return self.n_ctx - self.safety_buffer_tokens

    @property
    def effective_history_limit(self) -> int:
        """Эффективный лимит истории с учетом RAG."""
        if self.enable_rag:
            return min(
                self.max_history_tokens,
                self.available_context_tokens - self.rag_max_context - self.max_response_tokens
            )
        return min(self.max_history_tokens, self.available_context_tokens - self.max_response_tokens)

    @property
    def context_distribution(self) -> dict:
        """Текущее распределение контекста для отладки."""
        return {
            "total_context": self.n_ctx,
            "history_tokens": self.max_history_tokens,
            "response_tokens": self.max_response_tokens,
            "rag_tokens": self.rag_max_context if self.enable_rag else 0,
            "safety_buffer": self.safety_buffer_tokens,
            "allocated_total": (
                self.max_history_tokens +
                self.max_response_tokens +
                (self.rag_max_context if self.enable_rag else 0) +
                self.safety_buffer_tokens
            ),
            "remaining": self.n_ctx - (
                self.max_history_tokens +
                self.max_response_tokens +
                (self.rag_max_context if self.enable_rag else 0) +
                self.safety_buffer_tokens
            )
        }

    def get_optimal_device(self) -> str:
        """Определить оптимальное устройство для вычислений."""
        if self.compute_device == "cpu":
            return "cpu"
        elif self.compute_device == "cuda":
            return "cuda" if self._is_cuda_available() else "cpu"
        else:  # auto
            if self._is_cuda_available() and not self.force_cpu_embedding:
                return "cuda"
            return "cpu"

    def get_embedding_device(self) -> str:
        """Определить устройство для embedding вычислений."""
        if self.force_cpu_embedding:
            return "cpu"
        return self.get_optimal_device()

    def _is_cuda_available(self) -> bool:
        """Проверить доступность CUDA."""
        try:
            # Ленивый импорт torch только когда нужна диагностика
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_model_type(self) -> str:
        return self.model_type

    @property
    def device_info(self) -> dict:
        """Информация об устройствах для диагностики."""
        try:
            # Ленивый импорт torch только когда нужна диагностика
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            cuda_memory = []
            if cuda_available:
                for i in range(cuda_device_count):
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    cuda_memory.append(f"GPU {i}: {total_memory // 1024**3}GB")
        except ImportError:
            cuda_available = False
            cuda_device_count = 0
            cuda_memory = []

        return {
            "compute_device_setting": self.compute_device,
            "force_cpu_embedding": self.force_cpu_embedding,
            "optimal_device": self.get_optimal_device(),
            "embedding_device": self.get_embedding_device(),
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_memory": cuda_memory,
            "n_gpu_layers": self.n_gpu_layers,
        }


@lru_cache
def get_settings() -> Settings:
    """Получить настройки приложения (singleton)."""
    return Settings()
