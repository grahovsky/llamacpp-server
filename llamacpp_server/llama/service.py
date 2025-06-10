"""Сервис для работы с LLama.cpp в RAG-only режиме."""

import time
import uuid
from collections.abc import AsyncIterator

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
from ..retrieval.protocols import RAGServiceProtocol
from ..prompts.templates import SYSTEM_PROMPT
from .history_manager import ChatHistoryManager

logger = structlog.get_logger(__name__)


class LlamaService:
    """Сервис для работы с LLama моделью в RAG-only режиме."""

    def __init__(
        self,
        llama: Llama,
        rag_service: RAGServiceProtocol = None
    ) -> None:
        """Инициализация сервиса."""
        self._llama = llama
        self._rag_service = rag_service
        settings = get_settings()
        
        if not rag_service:
            raise ValueError("RAG сервис обязателен для RAG-only системы")
        
        self.history_manager = ChatHistoryManager(
            llama_model=llama,
            max_tokens=settings.max_history_tokens,
            reserve_tokens=settings.max_response_tokens
        )
        
        logger.info("LlamaService инициализирован в RAG-only режиме")

    async def chat_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Генерация chat completion с обязательным RAG."""
        logger.info("🧠 RAG-only chat completion", model=request.model)

        # Извлекаем последний пользовательский запрос
        user_query = self._extract_user_query(request.messages)
        
        if not user_query:
            logger.warning("Нет пользовательского запроса")
            return self._create_error_response("Не найден пользовательский запрос")

        logger.info("Обработка запроса", query_preview=user_query[:100])

        try:
            # Создаем RAG промпт для любого запроса
            rag_prompt = await self._rag_service.create_rag_prompt(user_query)
            
            # Создаем новые сообщения с системным промптом и RAG промптом
            rag_messages = [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=rag_prompt)
            ]

            # Создаем новый запрос с RAG сообщениями
            rag_request = ChatCompletionRequest(
                messages=rag_messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=request.stream
            )

            logger.info("✅ RAG промпт создан", 
                       original_len=len(user_query),
                       rag_len=len(rag_prompt))

            # Обрабатываем запрос
            return await self._process_completion(rag_request)

        except Exception as e:
            logger.error("❌ Ошибка RAG обработки", error=str(e), exc_info=True)
            return self._create_error_response(f"Ошибка обработки: {str(e)}")

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг chat completion с обязательным RAG."""
        logger.info("🧠 RAG-only streaming chat completion", model=request.model)

        # Извлекаем пользовательский запрос
        user_query = self._extract_user_query(request.messages)
        
        if not user_query:
            logger.warning("Нет пользовательского запроса для стриминга")
            yield self._create_stream_error("Не найден пользовательский запрос")
            return

        try:
            # Создаем RAG промпт
            rag_prompt = await self._rag_service.create_rag_prompt(user_query)
            
            # Создаем RAG сообщения
            rag_messages = [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=rag_prompt)
            ]

            # Создаем RAG запрос
            rag_request = ChatCompletionRequest(
                messages=rag_messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=True
            )

            logger.info("✅ RAG стриминг промпт создан")

            # Стримим ответ
            async for chunk in self._process_completion_stream(rag_request):
                yield chunk

        except Exception as e:
            logger.error("❌ Ошибка RAG стриминга", error=str(e), exc_info=True)
            yield self._create_stream_error(f"Ошибка обработки: {str(e)}")

    async def text_completion(self, request: TextCompletionRequest) -> CompletionResponse:
        """Text completion с обязательным RAG."""
        logger.info("🧠 RAG-only text completion", model=request.model)

        if not request.prompt:
            return self._create_error_response("Пустой промпт")

        try:
            # Создаем RAG промпт
            rag_prompt = await self._rag_service.create_rag_prompt(request.prompt)
            
            # Создаем RAG запрос
            rag_request = TextCompletionRequest(
                prompt=rag_prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=request.stream
            )

            logger.info("✅ RAG text completion промпт создан")

            # Обрабатываем запрос
            return await self._process_text_completion(rag_request)

        except Exception as e:
            logger.error("❌ Ошибка RAG text completion", error=str(e), exc_info=True)
            return self._create_error_response(f"Ошибка обработки: {str(e)}")

    async def text_completion_stream(
        self, request: TextCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг text completion с обязательным RAG."""
        logger.info("🧠 RAG-only streaming text completion", model=request.model)

        if not request.prompt:
            yield self._create_stream_error("Пустой промпт")
            return

        try:
            # Создаем RAG промпт
            rag_prompt = await self._rag_service.create_rag_prompt(request.prompt)
            
            # Создаем RAG запрос
            rag_request = TextCompletionRequest(
                prompt=rag_prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=True
            )

            logger.info("✅ RAG streaming text completion промпт создан")

            # Стримим ответ
            async for chunk in self._process_text_completion_stream(rag_request):
                yield chunk

        except Exception as e:
            logger.error("❌ Ошибка RAG text completion стриминга", error=str(e), exc_info=True)
            yield self._create_stream_error(f"Ошибка обработки: {str(e)}")

    def _extract_user_query(self, messages: list[ChatMessage]) -> str:
        """Извлечь последний пользовательский запрос."""
        for msg in reversed(messages):
            if msg.role == "user" and msg.content.strip():
                return msg.content.strip()
        return ""

    async def _process_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Обработка completion запроса."""
        # Форматируем для llama.cpp
        formatted_prompt = self._format_chat_messages(request.messages)
        
        logger.debug("Отправка в LLM", prompt_len=len(formatted_prompt))

        # Генерируем ответ
        response = self._llama.create_completion(
            prompt=formatted_prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 40,
            repeat_penalty=request.repeat_penalty or 1.1,
            seed=request.seed or -1,
            stream=False,
        )

        return self._create_completion_response(response, request.model, "chat.completion")

    async def _process_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[dict]:
        """Обработка streaming completion запроса."""
        formatted_prompt = self._format_chat_messages(request.messages)
        
        logger.debug("Отправка streaming в LLM", prompt_len=len(formatted_prompt))

        def create_stream():
            return self._llama.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 40,
                repeat_penalty=request.repeat_penalty or 1.1,
                seed=request.seed or -1,
                stream=True,
            )

        async def process_stream():
            stream = create_stream()
            
            for chunk in stream:
                if chunk and "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                        if content:
                            yield {
                                "id": f"chatcmpl-{uuid.uuid4()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                    
                    if choice.get("finish_reason"):
                        yield {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": choice["finish_reason"],
                                }
                            ],
                        }

        async for chunk in process_stream():
            yield chunk

    async def _process_text_completion(self, request: TextCompletionRequest) -> CompletionResponse:
        """Обработка text completion запроса."""
        logger.debug("Отправка text completion в LLM", prompt_len=len(request.prompt))

        response = self._llama.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 40,
            repeat_penalty=request.repeat_penalty or 1.1,
            seed=request.seed or -1,
            stream=False,
        )

        return self._create_completion_response(response, request.model, "text_completion")

    async def _process_text_completion_stream(self, request: TextCompletionRequest) -> AsyncIterator[dict]:
        """Обработка streaming text completion запроса."""
        logger.debug("Отправка streaming text completion в LLM", prompt_len=len(request.prompt))

        def create_stream():
            return self._llama.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 40,
                repeat_penalty=request.repeat_penalty or 1.1,
                seed=request.seed or -1,
                stream=True,
            )

        async def process_stream():
            stream = create_stream()
            
            for chunk in stream:
                if chunk and "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "text" in choice:
                        text = choice["text"]
                        if text:
                            yield {
                                "id": f"cmpl-{uuid.uuid4()}",
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": text,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                    
                    if choice.get("finish_reason"):
                        yield {
                            "id": f"cmpl-{uuid.uuid4()}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": "",
                                    "finish_reason": choice["finish_reason"],
                                }
                            ],
                        }

        async for chunk in process_stream():
            yield chunk

    async def is_ready(self) -> bool:
        """Проверка готовности сервиса."""
        return self._llama is not None and self._rag_service is not None

    def _format_chat_messages(self, messages: list[ChatMessage]) -> str:
        """Форматирование сообщений для llama.cpp."""
        formatted_parts = []
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        formatted_parts.append("Assistant:")
        return "\n\n".join(formatted_parts)

    def _create_completion_response(
        self, response: dict, model: str, object_type: str
    ) -> CompletionResponse:
        """Создание ответа completion."""
        if not response or "choices" not in response:
            return self._create_error_response("Пустой ответ от модели")

        choice_data = response["choices"][0]
        
        # Определяем контент в зависимости от типа
        if object_type == "chat.completion":
            content = choice_data.get("message", {}).get("content", "")
        else:
            content = choice_data.get("text", "")

        choice = Choice(
            index=0,
            message=ChatMessage(role="assistant", content=content),
            finish_reason=choice_data.get("finish_reason", "stop"),
        )

        usage = Usage(
            prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
            total_tokens=response.get("usage", {}).get("total_tokens", 0),
        )

        return CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object=object_type,
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
        )

    def _create_error_response(self, error_message: str) -> CompletionResponse:
        """Создание ответа с ошибкой."""
        choice = Choice(
            index=0,
            message=ChatMessage(role="assistant", content=f"Ошибка: {error_message}"),
            finish_reason="stop",
        )

        return CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model="llama-cpp",
            choices=[choice],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def _create_stream_error(self, error_message: str) -> dict:
        """Создание streaming ошибки."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "llama-cpp",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Ошибка: {error_message}"},
                    "finish_reason": "stop",
                }
            ],
        } 