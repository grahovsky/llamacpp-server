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
        "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    )
    
    # === Параметры модели ===
    n_ctx: Annotated[int, Field(ge=1, description="Размер контекста")] = 4096
    n_batch: Annotated[int, Field(ge=1, description="Размер батча")] = 512
    n_threads: Annotated[int, Field(ge=1, description="Количество потоков")] = 8
    n_gpu_layers: Annotated[int, Field(ge=0, description="Количество слоев на GPU")] = 30
    main_gpu: Annotated[int, Field(ge=0, description="Основная GPU для вычислений")] = 0
    tensor_split: Annotated[str | None, Field(description="Разделение тензоров между GPU (через запятую)")] = None
    
    # === Управление историей чата ===
    max_history_tokens: Annotated[int, Field(ge=1, description="Максимум токенов в истории чата")] = 3000
    context_reserve_tokens: Annotated[int, Field(ge=1, description="Резерв токенов для ответа")] = 1000
    
    # === Сервер ===
    host: Annotated[str, Field(description="Хост сервера")] = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535, description="Порт сервера")] = 8090
    
    # === Дополнительные настройки ===
    verbose: Annotated[bool, Field(description="Подробный вывод")] = False
    use_mmap: Annotated[bool, Field(description="Использовать memory mapping")] = True
    use_mlock: Annotated[bool, Field(description="Использовать memory locking")] = False
    chat_format: Annotated[str, Field(description="Формат чата")] = "llama-2"
    temperature: Annotated[float, Field(ge=0.0, le=2.0, description="Температура")] = 0.7
    
    # === Логирование ===
    log_level: Annotated[str, Field(description="Уровень логирования")] = "DEBUG"
    
    # === Режим разработки ===
    dev_mode: Annotated[bool, Field(description="Режим разработки (без модели)")] = False
    
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
        if not v.exists():
            raise ValueError(f"Файл модели не найден: {v}")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (singleton)."""
    return Settings() 