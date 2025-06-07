"""Сервис для работы с LLama.cpp."""

import time
import uuid
from typing import AsyncIterator

import structlog
from llama_cpp import Llama

from ..config.settings import get_settings
from ..domain.models import (
    ChatCompletionRequest,
    ChatMessage,
    Choice,
    CompletionResponse,
    TextCompletionRequest,
    Usage,
)
from .history_manager import ChatHistoryManager


logger = structlog.get_logger(__name__)


class LlamaService:
    """Сервис для работы с LLama моделью."""
    
    def __init__(self, llama: Llama) -> None:
        self._llama = llama
        settings = get_settings()
        self._history_manager = ChatHistoryManager(
            llama_model=llama,
            max_tokens=settings.max_history_tokens,
            reserve_tokens=settings.context_reserve_tokens
        )
        
    async def chat_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Генерация chat completion."""
        logger.info("Обработка chat completion запроса", model=request.model)
        
        # Подготавливаем сообщения с учетом лимитов контекста
        messages_dict = [msg.dict() for msg in request.messages]
        trimmed_messages = self._history_manager.prepare_messages_for_completion(messages_dict)
        
        # Форматируем сообщения в промпт
        formatted_prompt = self._format_chat_messages([
            ChatMessage(**msg) for msg in trimmed_messages
        ])
        
        # Генерируем ответ
        if hasattr(self._llama, 'create_chat_completion'):
            # Используем chat completion если доступен
            response = self._llama.create_chat_completion(
                messages=trimmed_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=False,
            )
        else:
            # Fallback на обычную генерацию
            response = self._llama.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=False,
            )
        
        # Формируем ответ в OpenAI формате
        return self._create_completion_response(response, request.model, "chat.completion")
    
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг chat completion."""
        logger.info("Обработка streaming chat completion запроса", model=request.model)
        
        # Подготавливаем сообщения с учетом лимитов контекста
        messages_dict = [msg.dict() for msg in request.messages]
        trimmed_messages = self._history_manager.prepare_messages_for_completion(messages_dict)
        
        formatted_prompt = self._format_chat_messages([
            ChatMessage(**msg) for msg in trimmed_messages
        ])
        
        # Запускаем в thread pool чтобы не блокировать event loop
        import asyncio
        import concurrent.futures
        
        def create_stream():
            return self._llama.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=True,
            )
        
        # Обрабатываем stream в executor
        loop = asyncio.get_event_loop()
        
        async def process_stream():
            # Получаем stream
            stream = create_stream()
            
            # Если это list (mock), обрабатываем с задержками
            if isinstance(stream, list):
                for chunk in stream:
                    await asyncio.sleep(0.2)  # Эмулируем задержку генерации
                    
                    # Приводим к стандартному формату
                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        if "delta" in choice and choice["delta"]:
                            yield chunk
                        elif "text" in choice:
                            yield {
                                "choices": [{
                                    "delta": {"content": choice["text"]},
                                    "finish_reason": choice.get("finish_reason")
                                }]
                            }
                        else:
                            yield chunk
                    else:
                        yield chunk
            else:
                # Реальный stream от llama.cpp
                def get_next_chunk(stream_iter):
                    try:
                        return next(stream_iter)
                    except StopIteration:
                        return None
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    stream_iter = iter(stream)
                    
                    while True:
                        # Получаем следующий чанк в thread pool
                        chunk = await loop.run_in_executor(executor, get_next_chunk, stream_iter)
                        
                        if chunk is None:
                            break
                        
                        # Приводим к стандартному формату
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            if "delta" in choice and choice["delta"]:
                                yield chunk
                            elif "text" in choice:
                                yield {
                                    "choices": [{
                                        "delta": {"content": choice["text"]},
                                        "finish_reason": choice.get("finish_reason")
                                    }]
                                }
                            else:
                                yield chunk
                        else:
                            yield chunk
        
        async for chunk in process_stream():
            yield chunk
    
    async def text_completion(self, request: TextCompletionRequest) -> CompletionResponse:
        """Генерация text completion."""
        logger.info("Обработка text completion запроса", model=request.model)
        
        response = self._llama.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            seed=request.seed,
            stream=False,
        )
        
        return self._create_completion_response(response, request.model, "text_completion")
    
    async def text_completion_stream(
        self, request: TextCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг text completion."""
        logger.info("Обработка streaming text completion запроса", model=request.model)
        
        # Запускаем в thread pool чтобы не блокировать event loop
        import asyncio
        import concurrent.futures
        
        def create_stream():
            return self._llama.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=True,
            )
        
        # Обрабатываем stream в executor
        loop = asyncio.get_event_loop()
        
        async def process_stream():
            # Получаем stream
            stream = create_stream()
            
            # Если это list (mock), обрабатываем с задержками
            if isinstance(stream, list):
                for chunk in stream:
                    await asyncio.sleep(0.2)  # Эмулируем задержку генерации
                    
                    # Стандартизируем формат чанков
                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        if "delta" in choice and choice["delta"]:
                            yield chunk
                        elif "text" in choice:
                            yield {
                                "choices": [{
                                    "delta": {"content": choice["text"]},
                                    "finish_reason": choice.get("finish_reason"),
                                    "text": choice["text"]
                                }]
                            }
                        else:
                            yield chunk
                    else:
                        yield chunk
            else:
                # Реальный stream от llama.cpp
                def get_next_chunk(stream_iter):
                    try:
                        return next(stream_iter)
                    except StopIteration:
                        return None
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    stream_iter = iter(stream)
                    
                    while True:
                        # Получаем следующий чанк в thread pool
                        chunk = await loop.run_in_executor(executor, get_next_chunk, stream_iter)
                        
                        if chunk is None:
                            break
                        
                        # Стандартизируем формат чанков
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            if "delta" in choice and choice["delta"]:
                                yield chunk
                            elif "text" in choice:
                                yield {
                                    "choices": [{
                                        "delta": {"content": choice["text"]},
                                        "finish_reason": choice.get("finish_reason"),
                                        "text": choice["text"]
                                    }]
                                }
                            else:
                                yield chunk
                        else:
                            yield chunk
        
        async for chunk in process_stream():
            yield chunk
    
    async def is_ready(self) -> bool:
        """Проверка готовности модели."""
        return True  # LLama модель готова после инициализации
    
    def _format_chat_messages(self, messages: list[ChatMessage]) -> str:
        """Форматирование сообщений в промпт."""
        # Простое форматирование для демонстрации
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
        
        formatted += "Assistant: "
        return formatted
    
    def _create_completion_response(
        self, response: dict, model: str, object_type: str
    ) -> CompletionResponse:
        """Создание ответа в OpenAI формате."""
        choice = response["choices"][0]
        
        # Обработка разных типов ответов
        if object_type == "chat.completion" and "message" in choice:
            # Chat completion ответ
            message_content = choice["message"]["content"]
            choice_obj = Choice(
                index=0,
                message=ChatMessage(role="assistant", content=message_content),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        else:
            # Text completion ответ
            text_content = choice.get("text", "")
            choice_obj = Choice(
                index=0,
                text=text_content,
                message=(
                    ChatMessage(role="assistant", content=text_content)
                    if object_type == "chat.completion"
                    else None
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            object=object_type,
            created=int(time.time()),
            model=model,
            choices=[choice_obj],
            usage=Usage(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
                total_tokens=response["usage"]["total_tokens"],
            ),
        ) 