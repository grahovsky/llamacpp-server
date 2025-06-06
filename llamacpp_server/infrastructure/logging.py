"""Настройка логирования с structlog."""

import logging
import sys
from typing import Any, Dict

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Настройка структурированного логирования."""
    
    # Конфигурация timestamper
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    # Конфигурация для красивых логов в dev режиме
    if log_level in ["DEBUG", "INFO"]:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.JSONRenderer(),
        ]
    
    # Настройка structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    
    # Настройка стандартного логгера Python
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    
    # Подавляем слишком болтливые библиотеки
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("llama_cpp").setLevel(logging.WARNING) 