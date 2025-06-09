"""Fallback роутер для обработки неизвестных эндпоинтов."""

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def catch_all(request: Request, full_path: str):
    """Обработчик для всех неизвестных эндпоинтов."""
    method = request.method
    path = request.url.path

    logger.warning(
        "Неизвестный эндпоинт",
        method=method,
        path=path,
        headers=dict(request.headers),
        query_params=dict(request.query_params)
    )

    # Подсказки для популярных API
    suggestions = []

    if "/api/" in path:
        suggestions.extend([
            "/api/tags - список моделей (Open WebUI)",
            "/api/show - информация о модели (Open WebUI)",
            "/api/version - версия API (Open WebUI)"
        ])

    if "/ollama/" in path:
        suggestions.extend([
            "/ollama/api/tags - список моделей (Ollama)",
            "/ollama/api/chat - чат completion (Ollama)",
            "/ollama/api/generate - генерация текста (Ollama)"
        ])

    if "/v1/" in path:
        suggestions.extend([
            "/v1/models - список моделей (OpenAI)",
            "/v1/chat/completions - чат completion (OpenAI)",
            "/v1/completions - text completion (OpenAI)"
        ])

    if not suggestions:
        suggestions = [
            "/health - проверка состояния",
            "/v1/models - OpenAI API",
            "/api/tags - Open WebUI API",
            "/ollama/api/tags - Ollama API",
            "/docs - документация API"
        ]

    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"Endpoint '{method} {path}' not found",
                "type": "not_found_error",
                "code": "endpoint_not_found",
                "suggestions": suggestions
            }
        }
    )
