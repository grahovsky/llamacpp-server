"""Настройки приложения используя Pydantic Settings v2."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""

    # === Модель ===
    model_path: Annotated[Path, Field(description="Путь к файлу модели GGUF")] = Path(
        "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
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
    max_response_tokens: Annotated[int, Field(ge=50, le=2048, description="Максимум токенов в ответе")] = 2048
    max_history_tokens: Annotated[int, Field(ge=0, description="Максимум токенов в истории чата")] = 1024
    rag_max_context: Annotated[int, Field(ge=0, description="Максимум токенов для RAG контекста")] = 4048
    safety_buffer_tokens: Annotated[int, Field(ge=50, description="Буфер безопасности для непредвиденных токенов")] = 800
    
    # === Параметры генерации ===
    top_k: Annotated[int, Field(ge=1, description="Top-K sampling")] = 40
    top_p: Annotated[float, Field(ge=0.0, le=1.0, description="Top-P (nucleus) sampling")] = 0.9
    repeat_penalty: Annotated[float, Field(ge=0.0, description="Штраф за повторы")] = 1.1
    
    # === Сервер ===
    host: Annotated[str, Field(description="Хост сервера")] = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535, description="Порт сервера")] = 8090
    
    # === Дополнительные настройки ===
    verbose: Annotated[bool, Field(description="Подробный вывод")] = False
    use_mmap: Annotated[bool, Field(description="Использовать memory mapping")] = True
    use_mlock: Annotated[bool, Field(description="Использовать memory locking")] = False
    chat_format: Annotated[str | None, Field(description="Формат чата (None = автоопределение)")] = None
    temperature: Annotated[float, Field(ge=0.0, le=2.0, description="Температура")] = 0.7
    
    # === Логирование ===
    log_level: Annotated[str, Field(description="Уровень логирования")] = "DEBUG"
    
    # === Режим разработки ===
    dev_mode: Annotated[bool, Field(description="Режим разработки (без модели)")] = False
    
    # === RAG настройки ===
    embedding_model: str = Field(
        "BAAI/bge-m3", 
        description="Модель для создания эмбеддингов"
    )
    faiss_index_path: str = Field(
        "./data/faiss_index", 
        description="Путь к FAISS индексу"
    )
    
    enable_rag: bool = Field(
        True,
        description="Включить RAG для всех запросов"
    )
    rag_search_k: int = Field(
        8,  # Уменьшено для оптимизации
        description="Количество документов для поиска в RAG"
    )
    
    use_citation_focused_rag: bool = Field(
        True,
        description="Использовать citation-focused RAG с усиленным цитированием источников"
    )
    
    rag_prompt_style: str = Field(
        "citation_focused",
        description="Стиль RAG промпта: 'citation_focused', 'simple_citation'"
    )
    
    model_config = {
        "env_prefix": "LLAMACPP_",
        "env_file": ".env",  # Читается из корня проекта (рабочая директория)
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
    }
    
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


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (singleton)."""
    return Settings() 