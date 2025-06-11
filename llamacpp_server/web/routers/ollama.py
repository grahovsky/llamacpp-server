"""Ollama-совместимый API роутер."""

import os
import time
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...config import get_settings
from ...domain.models import ChatMessage
from ...domain.protocols import LlamaServiceProtocol

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
        "Валидация модели",
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
            "Модель не прошла валидацию",
            requested=requested_model,
            normalized=normalized,
            available=available,
            variants_checked=len(variants)
        )

    return result


@router.get("/api/tags")
async def ollama_list_models():
    """Ollama: Список доступных моделей."""
    settings = get_settings()
    model_name = get_available_model_name()
    # Клиенты автоматически добавляют :latest, поэтому возвращаем сразу с тегом
    model_with_tag = f"{model_name}:latest"

    return {
        "models": [
            {
                "name": model_with_tag,
                "model": model_with_tag,
                "modified_at": "2024-01-01T00:00:00.000Z",
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


@router.get("/api/ps")
async def ollama_list_running(
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Ollama: Список запущенных моделей."""
    settings = get_settings()
    model_name = get_available_model_name()

    is_ready = await llama_service.is_ready()
    if is_ready:
        return {
            "models": [
                {
                    "name": model_name,
                    "model": model_name,
                    "size": os.path.getsize(settings.model_path) if settings.model_path.exists() else 0,
                    "digest": "sha256:dummy",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M"
                    },
                    "expires_at": "0001-01-01T00:00:00Z",
                    "size_vram": 0
                }
            ]
        }
    else:
        return {"models": []}


@router.get("/api/version")
async def ollama_version():
    """Ollama: Версия API."""
    return {"version": "0.1.0"}


@router.post("/api/show")
async def ollama_show_model(request_data: dict):
    """Ollama: Информация о модели."""
    settings = get_settings()
    model_name = request_data.get("name", "")

    if not validate_model_name(model_name):
        available_model = get_available_model_name()
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: '{available_model}'"
        )

    return {
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


@router.post("/api/generate")
async def ollama_generate(
    request_data: dict,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Ollama: Генерация текста."""
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
            max_tokens=get_settings().get_effective_max_tokens(request_data.get("max_tokens")),
            temperature=request_data.get("temperature", get_settings().temperature),
            top_p=request_data.get("top_p", get_settings().top_p),
            top_k=request_data.get("top_k", get_settings().top_k),
            repeat_penalty=request_data.get("repeat_penalty", get_settings().repeat_penalty),
            seed=request_data.get("seed", None),
            stream=stream
        )

        if stream:
            return StreamingResponse(
                _ollama_generate_stream(llama_service, completion_request, model_name),
                media_type="application/x-ndjson",
                headers={
                    "Content-Type": "application/x-ndjson; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "X-Accel-Buffering": "no"  # Отключаем буферизацию nginx
                }
            )
        else:
            response = await llama_service.text_completion(completion_request)
            response_text = response.choices[0].text if response.choices else ""

            # Проверяем, что ответ не пустой
            if not response_text.strip():
                logger.warning("Пустой ответ для ollama generate", model=model_name, prompt=prompt[:100])
                response_text = "Извините, не смог сгенерировать ответ. Попробуйте переформулировать запрос."

            return {
                "model": model_name,
                "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                "response": response_text,
                "done": True,
                "context": [],
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
                "prompt_eval_duration": 500000000,
                "eval_count": response.usage.completion_tokens if response.usage else 0,
                "eval_duration": 400000000
            }

    except Exception as e:
        logger.error("Ошибка Ollama generate", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/api/chat")
async def ollama_chat(
    request_data: dict,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Ollama: Чат completion."""
    model_name = request_data.get("model", "")
    messages = request_data.get("messages", [])
    stream = request_data.get("stream", False)

    logger.info("🔍 Входящий запрос ollama chat", 
                model=model_name, 
                messages_count=len(messages),
                stream=stream,
                first_message=messages[0] if messages else None)

    if not validate_model_name(model_name):
        available_model = get_available_model_name()
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: '{available_model}'"
        )

    if not messages:
        raise HTTPException(status_code=400, detail="Messages are required")

    try:
        from ...domain.models import ChatCompletionRequest

        chat_messages = [ChatMessage(**msg) for msg in messages]
        completion_request = ChatCompletionRequest(
            messages=chat_messages,
            model=model_name,
            max_tokens=get_settings().get_effective_max_tokens(request_data.get("max_tokens")),
            temperature=request_data.get("temperature", get_settings().temperature),
            top_p=request_data.get("top_p", get_settings().top_p),
            top_k=request_data.get("top_k", get_settings().top_k),
            repeat_penalty=request_data.get("repeat_penalty", get_settings().repeat_penalty),
            seed=request_data.get("seed", None),
            stream=stream
        )

        if stream:
            return StreamingResponse(
                _ollama_chat_stream(llama_service, completion_request, model_name),
                media_type="application/x-ndjson",
                headers={
                    "Content-Type": "application/x-ndjson; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "X-Accel-Buffering": "no"  # Отключаем буферизацию nginx
                }
            )
        else:
            logger.info("🧠 Запрос не-стриминг чат генерации", model=model_name)
            response = await llama_service.chat_completion(completion_request)
            response_text = response.choices[0].message.content if response.choices else ""

            # Проверяем, что ответ не пустой
            if not response_text or not response_text.strip():
                logger.warning("⚠️ Пустой ответ для ollama chat", 
                             model=model_name, 
                             messages_preview=str(messages)[:200])
                response_text = "Извините, не смог сгенерировать ответ на ваш запрос. Попробуйте переформулировать вопрос."

            result = {
                "model": model_name,
                "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                "message": {
                    "role": "assistant",
                    "content": response_text,
                    "images": None
                },
                "done": True,
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
                "prompt_eval_duration": 500000000,
                "eval_count": response.usage.completion_tokens if response.usage else 0,
                "eval_duration": 400000000
            }

            logger.info("✅ Ollama chat ответ готов", 
                       content_length=len(response_text),
                       preview=response_text[:100])
            
            return result

    except Exception as e:
        logger.error("❌ Ошибка Ollama chat", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


async def _ollama_generate_stream(
    llama_service: LlamaServiceProtocol,
    request: "TextCompletionRequest",
    model_name: str
) -> AsyncIterator[str]:
    """Стриминг для Ollama generate."""
    import json

    logger.info("🔄 Запуск ollama generate stream", model=model_name)
    
    try:
        chunk_count = 0
        total_content = ""
        
        async for chunk in llama_service.text_completion_stream(request):
            chunk_count += 1
            
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                content = choice.get("text", "") or choice.get("delta", {}).get("content", "")
                
                if content:  # Отправляем только если есть контент
                    total_content += content
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
        
        logger.info("✅ Ollama generate stream завершен", 
                   chunks=chunk_count, 
                   total_length=len(total_content))
        
    except Exception as e:
        logger.error("❌ Ошибка ollama generate stream", error=str(e), exc_info=True)
        # Отправляем ошибку в формате Ollama
        error_chunk = {
            "model": model_name,
            "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
            "response": f"Ошибка генерации: {str(e)}",
            "done": True,
            "error": str(e)
        }
        yield f"{json.dumps(error_chunk, ensure_ascii=False)}\n"


async def _ollama_chat_stream(
    llama_service: LlamaServiceProtocol,
    request: "ChatCompletionRequest",
    model_name: str
) -> AsyncIterator[str]:
    """Стриминг для Ollama chat."""
    import json

    logger.info("🔄 Запуск ollama chat stream", model=model_name)
    
    try:
        chunk_count = 0
        total_content = ""
        
        logger.info("🔄 Начинаем итерацию по chat_completion_stream")
        
        async for chunk in llama_service.chat_completion_stream(request):
            chunk_count += 1
            logger.debug(f"📦 Получен chunk #{chunk_count}", chunk_keys=list(chunk.keys()) if chunk else None)
            
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                logger.debug(f"💬 Content из chunk: {repr(content)}")

                if content:  # Отправляем только если есть контент
                    total_content += content
                    response_chunk = {
                        "model": model_name,
                        "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "images": None
                        },
                        "done": False
                    }
                    chunk_json = f"{json.dumps(response_chunk, ensure_ascii=False)}\n"
                    logger.debug(f"📤 Отправляем chunk: {chunk_json[:100]}...")
                    yield chunk_json

        # Финальный chunk
        final_chunk = {
            "model": model_name,
            "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
            "message": {
                "role": "assistant",
                "content": "",
                "images": None
            },
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 500000000,
            "eval_count": 0,
            "eval_duration": 400000000
        }
        final_json = f"{json.dumps(final_chunk, ensure_ascii=False)}\n"
        logger.debug("📤 Отправляем финальный chunk")
        yield final_json
        
        logger.debug("✅ Ollama chat stream завершен", 
                   chunks=chunk_count, 
                   total_length=len(total_content))
        
    except Exception as e:
        logger.error("❌ Ошибка ollama chat stream", error=str(e), exc_info=True)
        # Отправляем ошибку в формате Ollama
        error_chunk = {
            "model": model_name,
            "created_at": f"{time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}",
            "message": {
                "role": "assistant",
                "content": f"Ошибка генерации: {str(e)}",
                "images": None
            },
            "done": True,
            "error": str(e)
        }
        yield f"{json.dumps(error_chunk, ensure_ascii=False)}\n"
