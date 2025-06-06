"""Главное FastAPI приложение."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..infrastructure import Container, setup_logging
from .routers import health, openai, ollama, webui, fallback


logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan контекст для инициализации и очистки ресурсов."""
    logger.info("Запуск приложения")
    
    # Инициализация DI контейнера
    container = Container()
    
    # Создаем LLama инстанс
    from ..llama.factory import LlamaFactory
    factory = LlamaFactory()
    settings = container.config()
    llama_instance = await factory.create_llama(settings)
    
    # Переопределяем LLama инстанс в контейнере
    container.llama_instance.override(llama_instance)
    
    # Привязываем контейнер к приложению
    app.state.container = container
    
    logger.info("Приложение готово к работе")
    
    yield
    
    logger.info("Остановка приложения")


def create_app() -> FastAPI:
    """Создать FastAPI приложение."""
    
    # Настройка логирования
    setup_logging()
    
    app = FastAPI(
        title="LlamaCpp Server",
        description="OpenAI-compatible HTTP API server for llama.cpp",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        """Обработчик 404 ошибки."""
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Endpoint '{request.url.path}' not found",
                    "type": "not_found_error",
                    "code": "endpoint_not_found"
                }
            }
        )
    
    # Подключаем роутеры (порядок важен!)
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(openai.router, prefix="/v1", tags=["openai"])
    app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
    app.include_router(webui.router, tags=["webui"])
    
    # Fallback роутер должен быть последним!
    app.include_router(fallback.router, tags=["fallback"])
    
    return app 