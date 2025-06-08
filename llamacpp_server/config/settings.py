"""Настройки приложения используя Pydantic Settings v2."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
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
    n_ctx: Annotated[int, Field(ge=1, description="Размер контекста")] = 8192
    n_batch: Annotated[int, Field(ge=1, description="Размер батча")] = 512
    n_threads: Annotated[int, Field(ge=1, description="Количество потоков")] = 8
    n_gpu_layers: Annotated[int, Field(ge=0, description="Количество слоев на GPU")] = 30
    main_gpu: Annotated[int, Field(ge=0, description="Основная GPU для вычислений")] = 0
    tensor_split: Annotated[str | None, Field(description="Разделение тензоров между GPU (через запятую)")] = None
    
    # === Управление историей чата ===
    max_history_tokens: Annotated[int, Field(ge=1, description="Максимум токенов в истории чата")] = 3800  # Увеличиваем для RAG
    context_reserve_tokens: Annotated[int, Field(ge=1, description="Резерв токенов для ответа")] = 200   # Уменьшаем резерв
    
    # === Сервер ===
    host: Annotated[str, Field(description="Хост сервера")] = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535, description="Порт сервера")] = 8090
    
    # === Дополнительные настройки ===
    verbose: Annotated[bool, Field(description="Подробный вывод")] = False
    use_mmap: Annotated[bool, Field(description="Использовать memory mapping")] = True
    use_mlock: Annotated[bool, Field(description="Использовать memory locking")] = False
    chat_format: Annotated[str | None, Field(description="Формат чата (None = автоопределение)")] = None
    temperature: Annotated[float, Field(ge=0.0, le=2.0, description="Температура")] = 0.1
    
    # === Логирование ===
    log_level: Annotated[str, Field(description="Уровень логирования")] = "DEBUG"
    
    # === Режим разработки ===
    dev_mode: Annotated[bool, Field(description="Режим разработки (без модели)")] = False
    
    # RAG настройки
    embedding_model: str = Field(
        "BAAI/bge-m3", 
        description="Модель для создания эмбеддингов"
    )
    faiss_index_path: str = Field(
        "./data/faiss_index", 
        description="Путь к FAISS индексу"
    )
    max_context_length: int = Field(
        6000,  # Оптимальная длина для уменьшения обрезки
        description="Максимальная длина контекста для RAG"
    )

    enable_rag: bool = Field(
        True,  # Включен с улучшенным форматированием
        description="Включить RAG для всех запросов"
    )
    rag_search_k: int = Field(
        8,  # Уменьшаем для контроля размера контекста
        description="Количество документов для поиска в RAG"
    )
    
    use_citation_focused_rag: bool = Field(
        True,
        description="Использовать citation-focused RAG с усиленным цитированием источников"
    )
    
    rag_prompt_style: str = Field(
        "simple_citation",
        description="Стиль RAG промпта: 'strict_context', 'citation_focused', 'simple_citation'"
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


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (singleton)."""
    return Settings() 