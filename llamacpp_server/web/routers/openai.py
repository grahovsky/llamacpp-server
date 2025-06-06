"""OpenAI compatible API роутер."""

import time
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...domain.models import (
    ChatCompletionRequest,
    CompletionResponse,
    TextCompletionRequest,
)
from ...domain.protocols import LlamaServiceProtocol


router = APIRouter()
logger = structlog.get_logger(__name__)


async def get_llama_service(request: Request) -> LlamaServiceProtocol:
    """Получить LLama сервис из DI контейнера."""
    return request.app.state.container.llama_service()


@router.get("/models")
async def list_models():
    """Список доступных моделей."""
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-cpp",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llamacpp-server",
            }
        ],
    }


@router.post("/chat/completions", response_model=CompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service),
):
    """Chat completions endpoint."""
    logger.info("Получен запрос chat completions", model=request.model, stream=request.stream)
    
    try:
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(llama_service, request),
                media_type="text/plain",
            )
        else:
            response = await llama_service.chat_completion(request)
            return response
            
    except Exception as e:
        logger.error("Ошибка обработки chat completion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions", response_model=CompletionResponse)
async def text_completions(
    request: TextCompletionRequest,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service),
):
    """Text completions endpoint."""
    logger.info("Получен запрос text completions", model=request.model, stream=request.stream)
    
    try:
        if request.stream:
            return StreamingResponse(
                _stream_text_completion(llama_service, request),
                media_type="text/plain",
            )
        else:
            response = await llama_service.text_completion(request)
            return response
            
    except Exception as e:
        logger.error("Ошибка обработки text completion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat_completion(
    llama_service: LlamaServiceProtocol, request: ChatCompletionRequest
) -> AsyncIterator[str]:
    """Стриминг chat completion."""
    async for chunk in llama_service.chat_completion_stream(request):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_text_completion(
    llama_service: LlamaServiceProtocol, request: TextCompletionRequest
) -> AsyncIterator[str]:
    """Стриминг text completion."""
    async for chunk in llama_service.text_completion_stream(request):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n" 