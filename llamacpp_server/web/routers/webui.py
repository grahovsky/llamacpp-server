"""Open WebUI совместимый API роутер."""

import time
import os
import json
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...domain.models import ChatCompletionRequest, ChatMessage
from ...domain.protocols import LlamaServiceProtocol
from ...config import get_settings


router = APIRouter()
logger = structlog.get_logger(__name__)


async def get_llama_service(request: Request) -> LlamaServiceProtocol:
    """Получить LLama сервис из DI контейнера."""
    return request.app.state.container.llama_service()


def get_available_model_name() -> str:
    """Получить имя доступной модели."""
    settings = get_settings()
    return os.path.basename(str(settings.model_path)).replace(".gguf", "")


def validate_model_name(requested_model: str) -> bool:
    """Проверить доступность модели."""
    available = get_available_model_name()
    
    # Нормализуем запрошенную модель (убираем теги)
    normalized = requested_model.replace(":latest", "").replace(":stable", "")
    
    # Логируем для отладки
    logger.debug(
        "Валидация модели WebUI",
        requested=requested_model,
        normalized=normalized,
        available=available
    )
    
    # Проверяем различные варианты совпадения
    variants = [
        normalized == available,  # Точное совпадение нормализованной
        requested_model == available,  # Точное совпадение исходной
        normalized == "llama-cpp",
        normalized == "llamacpp",
        available.replace(".gguf", "") == normalized,  # Без расширения
        available.startswith(normalized),  # Начинается с
        normalized in available,  # Содержится в имени
    ]
    
    result = any(variants)
    
    if not result:
        logger.warning(
            "Модель не прошла валидацию WebUI",
            requested=requested_model,
            normalized=normalized,
            available=available,
            variants_checked=len(variants)
        )
    
    return result


@router.get("/api/tags")
async def get_tags():
    """Open WebUI: Список доступных моделей."""
    settings = get_settings()
    model_name = get_available_model_name()
    # Open WebUI автоматически добавляет :latest, поэтому возвращаем сразу с тегом
    model_with_tag = f"{model_name}:latest"
    
    return {
        "models": [
            {
                "name": model_with_tag,
                "model": model_with_tag,
                "modified_at": "2024-01-01T00:00:00Z",
                "size": os.path.getsize(settings.model_path) if settings.model_path.exists() else 0,
                "digest": "sha256:dummy",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_K_M"
                }
            }
        ]
    }


@router.get("/api/show")
async def show_model(name: str = "llama-cpp"):
    """Open WebUI: Информация о модели."""
    settings = get_settings()
    
    if not validate_model_name(name):
        available_model = get_available_model_name()
        raise HTTPException(
            status_code=400,
            detail=f"Model '{name}' was not found. Available model: '{available_model}'"
        )
    
    return {
        "license": "MIT",
        "modelfile": f"FROM {settings.model_path}",
        "parameters": f"temperature {settings.temperature}",
        "template": "{{ .System }}\n{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M"
        },
        "model_info": {
            "general.architecture": "llama",
            "general.file_type": 15,
            "general.parameter_count": 7000000000,
            "general.quantization_version": 2,
            "llama.attention.head_count": 32,
            "llama.block_count": 32,
            "llama.context_length": settings.n_ctx,
            "llama.embedding_length": 4096,
            "llama.feed_forward_length": 11008,
            "llama.rope.freq_base": 10000,
            "tokenizer.ggml.model": "llama"
        }
    }


@router.get("/api/version")
async def get_version():
    """Open WebUI: Версия API."""
    return {
        "version": "1.0.0",
        "api_version": "1.0.0"
    }


@router.post("/api/generate")
async def webui_generate(
    request_data: dict,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Open WebUI: Генерация текста."""
    model_name = request_data.get("model", "")
    prompt = request_data.get("prompt", "")
    stream = request_data.get("stream", False)
    
    if not validate_model_name(model_name):
        available_model = get_available_model_name()
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: '{available_model}'"
        )
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    try:
        from ...domain.models import TextCompletionRequest
        
        completion_request = TextCompletionRequest(
            prompt=prompt,
            model=model_name,
            max_tokens=request_data.get("max_tokens", get_settings().max_response_tokens),
            temperature=request_data.get("temperature", get_settings().temperature),
            top_p=request_data.get("top_p", get_settings().top_p),
            top_k=request_data.get("top_k", get_settings().top_k),
            repeat_penalty=request_data.get("repeat_penalty", get_settings().repeat_penalty),
            seed=request_data.get("seed", None),
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                _webui_generate_stream(llama_service, completion_request, model_name),
                media_type="application/x-ndjson",
                headers={
                    "Content-Type": "application/x-ndjson; charset=utf-8",
                    "Cache-Control": "no-cache", 
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Отключает nginx буферизацию
                }
            )
        else:
            response = await llama_service.text_completion(completion_request)
            response_text = response.choices[0].text if response.choices else ""
            
            return {
                "model": model_name,
                "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                "response": response_text,
                "done": True,
                "context": [],
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": response.usage.prompt_tokens,
                "prompt_eval_duration": 500000000,
                "eval_count": response.usage.completion_tokens,
                "eval_duration": 400000000
            }
        
    except Exception as e:
        logger.error("Ошибка WebUI generate", error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/api/chat")
async def webui_chat(
    request_data: dict,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Open WebUI: Чат completion."""
    model_name = request_data.get("model", "")
    messages = request_data.get("messages", [])
    stream = request_data.get("stream", False)
    
    if not validate_model_name(model_name):
        available_model = get_available_model_name()
        logger.error(
            "Модель не найдена в WebUI chat",
            requested=model_name,
            available=available_model
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: '{available_model}'"
        )
    
    if not messages:
        raise HTTPException(status_code=400, detail="Messages are required")
    
    try:
        chat_messages = [ChatMessage(**msg) for msg in messages]
        completion_request = ChatCompletionRequest(
            messages=chat_messages,
            model=model_name,
            max_tokens=request_data.get("max_tokens", get_settings().max_response_tokens),
            temperature=request_data.get("temperature", get_settings().temperature),
            top_p=request_data.get("top_p", get_settings().top_p),
            top_k=request_data.get("top_k", get_settings().top_k),
            repeat_penalty=request_data.get("repeat_penalty", get_settings().repeat_penalty),
            seed=request_data.get("seed", None),
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                _webui_chat_stream(llama_service, completion_request, model_name),
                media_type="application/x-ndjson",
                headers={
                    "Content-Type": "application/x-ndjson; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Отключает nginx буферизацию
                }
            )
        else:
            response = await llama_service.chat_completion(completion_request)
            response_text = response.choices[0].message.content if response.choices else ""
            
            return {
                "model": model_name,
                "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "done": True,
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": response.usage.prompt_tokens,
                "prompt_eval_duration": 500000000,
                "eval_count": response.usage.completion_tokens,
                "eval_duration": 400000000
            }
        
    except Exception as e:
        logger.error("Ошибка WebUI chat", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


async def _webui_generate_stream(
    llama_service: LlamaServiceProtocol, 
    request: "TextCompletionRequest", 
    model_name: str
) -> AsyncIterator[str]:
    """Стриминг для WebUI generate."""
    async for chunk in llama_service.text_completion_stream(request):
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            content = choice.get("text", "") or choice.get("delta", {}).get("content", "")
            
            response_chunk = {
                "model": model_name,
                "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                "response": content,
                "done": False
            }
            yield f"{json.dumps(response_chunk, ensure_ascii=False)}\n"
    
    # Финальный chunk
    final_chunk = {
        "model": model_name,
        "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
        "response": "",
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 500000000,
        "eval_count": 0,
        "eval_duration": 400000000
    }
    yield f"{json.dumps(final_chunk, ensure_ascii=False)}\n"


async def _webui_chat_stream(
    llama_service: LlamaServiceProtocol, 
    request: "ChatCompletionRequest", 
    model_name: str
) -> AsyncIterator[str]:
    """Стриминг для WebUI chat."""
    async for chunk in llama_service.chat_completion_stream(request):
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            
            response_chunk = {
                "model": model_name,
                "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": False
            }
            yield f"{json.dumps(response_chunk, ensure_ascii=False)}\n"
    
    # Финальный chunk
    final_chunk = {
        "model": model_name,
        "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
        "message": {
            "role": "assistant",
            "content": ""
        },
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 500000000,
        "eval_count": 0,
        "eval_duration": 400000000
    }
    yield f"{json.dumps(final_chunk, ensure_ascii=False)}\n" 