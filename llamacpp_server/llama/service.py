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
from ..prompts.protocols import PromptServiceProtocol
from .history_manager import ChatHistoryManager

logger = structlog.get_logger(__name__)


class LlamaService:
    """Сервис для работы с LLama моделью в RAG-only режиме."""

    def __init__(
        self,
        llama: Llama,
        rag_service: RAGServiceProtocol = None,
        prompt_service: PromptServiceProtocol = None
    ) -> None:
        """Инициализация сервиса."""
        self._llama = llama
        self._rag_service = rag_service
        self._prompt_service = prompt_service
        settings = get_settings()
        
        if not rag_service:
            raise ValueError("RAG сервис обязателен для RAG-only системы")
        
        if not prompt_service:
            raise ValueError("Prompt сервис обязателен")
        
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

        # Фильтрация запросов генерации заголовков - отвечаем без RAG
        if self._is_title_generation_request(user_query):
            logger.info("📝 Обнаружен запрос генерации заголовка, обрабатываем без RAG")
            return await self._process_simple_completion(request)

        try:
            # Создаем RAG промпт для любого запроса
            rag_prompt = await self._rag_service.create_rag_prompt(user_query)
            
            # RAG промпт уже содержит системный контекст в правильном формате
            # Создаем простое пользовательское сообщение с RAG промптом
            rag_messages = [
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
            
            # RAG промпт уже содержит системный контекст в правильном формате
            # Создаем простое пользовательское сообщение с RAG промптом
            rag_messages = [
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
        
        # Подготавливаем параметры
        params = {
            "prompt": formatted_prompt,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "top_k": request.top_k or 40,
            "repeat_penalty": request.repeat_penalty or 1.05,  # Уменьшаем для меньших повторений
            "seed": request.seed or -1,
            "stream": False,
            # Критически важно для Llama 3.1 - stop токены
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>", "Вопрос:", "useranswer", "Ответ:"],
        }
        
        logger.debug("🤖 LLM параметры", 
                    prompt_len=len(formatted_prompt),
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    repeat_penalty=params["repeat_penalty"],
                    seed=params["seed"])
        
        logger.debug("🔤 Промпт для LLM", prompt_preview=formatted_prompt[:500])

        # Генерируем ответ
        response = self._llama.create_completion(**params)
        
        logger.debug("🎯 LLM ответ", response_keys=list(response.keys()) if response else None)
        if response and "choices" in response:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", choice.get("text", ""))
            logger.debug("📝 Контент ответа", 
                        content_len=len(content), 
                        content_preview=content[:200],
                        finish_reason=choice.get("finish_reason"))

        return self._create_completion_response(response, request.model, "chat.completion")

    async def _process_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[dict]:
        """Обработка streaming completion запроса."""
        formatted_prompt = self._format_chat_messages(request.messages)
        
        # Подготавливаем параметры
        params = {
            "prompt": formatted_prompt,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "top_k": request.top_k or 40,
            "repeat_penalty": request.repeat_penalty or 1.05,  # Уменьшаем для меньших повторений
            "seed": request.seed or -1,
            "stream": True,
            # Критически важно для Llama 3.1 - stop токены
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>", "Вопрос:", "useranswer", "Ответ:"],
        }
        
        logger.debug("🤖 LLM streaming параметры", 
                    prompt_len=len(formatted_prompt),
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    repeat_penalty=params["repeat_penalty"],
                    seed=params["seed"])
        
        logger.debug("🔤 Streaming промпт для LLM", prompt_preview=formatted_prompt[:500])

        def create_stream():
            return self._llama.create_completion(**params)

        async def process_stream():
            stream = create_stream()
            total_content = ""
            chunk_count = 0
            
            logger.debug("🌀 Начало streaming генерации")
            
            for chunk in stream:
                chunk_count += 1
                logger.debug(f"📦 Chunk #{chunk_count}", chunk_keys=list(chunk.keys()) if chunk else None)
                
                if chunk and "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    logger.debug(f"🔍 Choice structure", choice_keys=list(choice.keys()), choice_preview=choice)
                    
                    # llama-cpp возвращает контент в "text", а не в "delta"
                    content = None
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                    elif "text" in choice:
                        content = choice["text"]
                    
                    if content:
                        total_content += content
                        logger.debug(f"💬 Chunk контент", content=repr(content))
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
                        logger.debug("🏁 Streaming завершен", 
                                   finish_reason=choice["finish_reason"],
                                   total_chunks=chunk_count,
                                   total_content_len=len(total_content))
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
            # Критически важно для Llama 3.1 - stop токены
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"],
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
                # Критически важно для Llama 3.1 - stop токены
                stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"],
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
        # Если только одно user сообщение и оно содержит структурированный RAG промпт
        # (содержит Llama 3.1 токены), используем его как есть
        if (len(messages) == 1 and 
            messages[0].role == "user" and 
            "<|begin_of_text|>" in messages[0].content):
            return messages[0].content
        
        # Иначе используем простое форматирование для обычных сообщений
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
        
        logger.debug("🔍 Raw choice data", choice_keys=list(choice_data.keys()), choice_data=choice_data)
        
        # Определяем контент в зависимости от типа
        if object_type == "chat.completion":
            # Пробуем разные варианты извлечения контента
            content = (
                choice_data.get("message", {}).get("content", "") or
                choice_data.get("text", "") or
                ""
            )
        else:
            content = choice_data.get("text", "")
            
        logger.debug("📝 Извлеченный контент", content_len=len(content), content_preview=content[:100])

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

    def _is_title_generation_request(self, query: str) -> bool:
        """Проверить, является ли запрос генерацией заголовка."""
        title_keywords = [
            "generate a concise",
            "word title",
            "summarizing the chat",
            "task:",
            "generate title",
            "create title",
            "make title",
            "title for",
            "chat history",
            "conversation title"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in title_keywords)

    async def _process_simple_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Простая обработка без RAG для заголовков."""
        try:
            # Создаем простой системный промпт для заголовков
            simple_messages = [
                ChatMessage(role="system", content="Ты помогаешь создавать краткие заголовки для чатов. Ответь кратко."),
                *request.messages
            ]
            
            simple_request = ChatCompletionRequest(
                messages=simple_messages,
                model=request.model,
                max_tokens=min(50, request.max_tokens or 50),  # Ограничиваем для заголовков
                temperature=0.3,  # Менее творческий для заголовков
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=False
            )
            
            return await self._process_completion(simple_request)
            
        except Exception as e:
            logger.error("❌ Ошибка простой обработки", error=str(e))
            return self._create_error_response(f"Ошибка: {str(e)}") 