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
from ..prompts.protocols import PromptServiceProtocol
from ..retrieval.protocols import RAGServiceProtocol
from .history_manager import ChatHistoryManager

logger = structlog.get_logger(__name__)


class LlamaService:
    """Сервис для работы с LLama моделью в RAG-only режиме."""

    def __init__(
        self,
        llama: Llama,
        rag_service: RAGServiceProtocol = None,
        prompt_service: PromptServiceProtocol = None,
        settings = None
    ) -> None:
        """Инициализация сервиса."""
        self._llama = llama
        self._rag_service = rag_service
        self._prompt_service = prompt_service

        # Если настройки не переданы, получаем их (для обратной совместимости)
        if settings is None:
            settings = get_settings()
        self._settings = settings

        if not rag_service:
            raise ValueError("RAG сервис обязателен для RAG-only системы")

        if not prompt_service:
            raise ValueError("Prompt сервис обязателен")

        self.history_manager = ChatHistoryManager(
            llama_model=llama,
            max_tokens=self._settings.max_history_tokens,
            reserve_tokens=self._settings.max_response_tokens
        )

        logger.info("LlamaService инициализирован в RAG-only режиме")

    async def chat_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Chat completion с обязательным RAG для всех запросов."""
        request_id = str(uuid.uuid4())[:8]  # Короткий ID для логов
        
        logger.info("🧠 Chat completion с обязательным RAG", 
                   model=request.model, request_id=request_id)

        # Извлекаем пользовательский запрос для анализа
        user_query = self._extract_user_query(request.messages)

        if not user_query:
            return self._create_error_response("Нет пользовательского запроса")

        logger.info("📝 Анализ запроса", 
                   query_preview=user_query[:100], request_id=request_id)

        # Проверяем, что RAG сервис доступен
        if not self._rag_service:
            logger.error("❌ RAG сервис недоступен - это обязательно для данного проекта")
            return self._create_error_response("RAG сервис недоступен")

        # Определяем тип запроса для выбора стратегии обработки
        if self._is_service_request(user_query):
            logger.info("🛠️ Служебный запрос - прямой proxy в LLM (без RAG)", request_id=request_id)
            return await self._process_service_proxy_completion(request, user_query, request_id)
        else:
            logger.info("🧠 Пользовательский запрос - используем полный RAG", request_id=request_id)
            return await self._process_full_rag_completion(request, user_query, request_id)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[dict]:
        """Стриминг chat completion с обязательным RAG для всех запросов."""
        logger.info("🧠 Streaming chat completion с обязательным RAG", model=request.model)

        # Извлекаем пользовательский запрос
        user_query = self._extract_user_query(request.messages)

        if not user_query:
            logger.warning("Нет пользовательского запроса для стриминга")
            yield self._create_stream_error("Не найден пользовательский запрос")
            return

        logger.info("📝 Анализ стриминг запроса", query_preview=user_query[:100])

        # Проверяем, что RAG сервис доступен
        if not self._rag_service:
            logger.error("❌ RAG сервис недоступен - это обязательно для данного проекта")
            yield self._create_stream_error("RAG сервис недоступен")
            return

        # Определяем тип запроса для выбора стратегии обработки
        if self._is_service_request(user_query):
            logger.info("🛠️ Служебный стриминг запрос - прямой proxy в LLM (без RAG)")
            async for chunk in self._process_service_proxy_completion_stream(request, user_query):
                yield chunk
        else:
            logger.info("🧠 Пользовательский стриминг запрос - используем полный RAG")
            async for chunk in self._process_full_rag_completion_stream(request, user_query):
                yield chunk

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

        # 🔍 ЛОГИРОВАНИЕ ФИНАЛЬНОГО ПРОМПТА
        logger.info("🔤 Финальный промпт для LLM", 
                   prompt_length=len(formatted_prompt),
                   starts_with_begin_of_text=formatted_prompt.startswith("<|begin_of_text|>"))
        
        # Выводим первые 1000 символов финального промпта
        logger.info("📋 Финальный промпт (первые 1000 символов):", 
                   final_prompt_preview=formatted_prompt[:1000])

        # Получаем централизованные RAG параметры с правильной обработкой max_tokens
        llm_params = self._settings.build_rag_params(
            is_title=False,
            max_tokens=request.max_tokens  # Передаем пользовательский лимит для валидации
        )

        # Подготавливаем параметры
        params = {
            "prompt": formatted_prompt,
            "stream": False,
            # Критически важно для Llama 3.1 - stop токены
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>", "Вопрос:", "useranswer", "Ответ:", "### Task:", "### Guidelines:", "### Output:", "### Examples:", "### Chat History:", "User:", "System:", "Assistant:"],
            **llm_params
        }

        logger.debug("🤖 LLM параметры",
                    prompt_len=len(formatted_prompt),
                    **{k: v for k, v in params.items() if k != "prompt" and k != "stop"})

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

        # Получаем централизованные RAG параметры с правильной обработкой max_tokens
        llm_params = self._settings.build_rag_params(
            is_title=False,
            max_tokens=request.max_tokens  # Передаем пользовательский лимит для валидации
        )

        # Подготавливаем параметры
        params = {
            "prompt": formatted_prompt,
            "stream": True,
            # Критически важно для Llama 3.1 - stop токены
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>", "Вопрос:", "useranswer", "Ответ:", "### Task:", "### Guidelines:", "### Output:", "### Examples:", "### Chat History:", "User:", "System:", "Assistant:"],
            **llm_params
        }

        logger.debug("🤖 LLM streaming параметры",
                    prompt_len=len(formatted_prompt),
                    **{k: v for k, v in params.items() if k != "prompt" and k != "stop"})

        logger.debug("🔤 Streaming промпт для LLM", prompt_preview=formatted_prompt[:500])

        # Запускаем streaming синхронно, но делаем его async через yield
        logger.info("🌊 Создание streaming генератора")
        stream = self._llama.create_completion(**params)
        total_content = ""
        chunk_count = 0

        # Простая защита от бесконечного стриминга
        max_tokens = self._settings.get_effective_max_tokens(request.max_tokens)
        token_count = 0

        logger.info("🌀 Начало streaming генерации", max_tokens=max_tokens)

        try:
            for chunk in stream:
                chunk_count += 1
                logger.debug(f"📦 Chunk #{chunk_count}", chunk_keys=list(chunk.keys()) if chunk else None)

                if chunk and "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    logger.debug("🔍 Choice structure", choice_keys=list(choice.keys()), choice_preview=choice)

                    # llama-cpp возвращает контент в "text", а не в "delta"
                    content = None
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                    elif "text" in choice:
                        content = choice["text"]

                    if content:
                        total_content += content
                        token_count += 1

                        # Простая защита от слишком длинного вывода
                        if token_count >= max_tokens:
                            logger.warning("🛑 Достигнут лимит токенов",
                                         max_tokens=max_tokens,
                                         total_chunks=chunk_count,
                                         total_content_len=len(total_content))
                            # Отправляем последний chunk с причиной остановки
                            yield {
                                "id": f"chatcmpl-{uuid.uuid4()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "length",
                                    }
                                ],
                            }
                            break

                        logger.debug("💬 Chunk контент", content=repr(content))
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
                        logger.info("🏁 Streaming завершен",
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
                        break

        except Exception as e:
            logger.error("❌ Ошибка в streaming генерации", error=str(e), exc_info=True)
            yield self._create_stream_error(f"Ошибка генерации: {str(e)}")

    async def _process_text_completion(self, request: TextCompletionRequest) -> CompletionResponse:
        """Обработка text completion запроса."""
        logger.debug("Отправка text completion в LLM", prompt_len=len(request.prompt))

        # Получаем централизованные RAG параметры (text completion с контекстом)
        llm_params = self._settings.merge_request_params(request, is_title=False)

        response = self._llama.create_completion(
            prompt=request.prompt,
            stream=False,
            # Критически важно для Llama 3.1 - stop токены
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"],
            **llm_params
        )

        return self._create_completion_response(response, request.model, "text_completion")

    async def _process_text_completion_stream(self, request: TextCompletionRequest) -> AsyncIterator[dict]:
        """Обработка streaming text completion запроса."""
        logger.debug("Отправка streaming text completion в LLM", prompt_len=len(request.prompt))

        # Получаем централизованные RAG параметры (text completion с контекстом)
        llm_params = self._settings.merge_request_params(request, is_title=False)

        # Запускаем streaming синхронно, но делаем его async через yield
        stream = self._llama.create_completion(
            prompt=request.prompt,
            stream=True,
            # Критически важно для Llama 3.1 - stop токены
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"],
            **llm_params
        )

        try:
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
                        break

        except Exception as e:
            logger.error("❌ Ошибка в text completion streaming", error=str(e), exc_info=True)
            yield self._create_stream_error(f"Ошибка генерации: {str(e)}")

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
            
            # ВАЖНО: убираем <|begin_of_text|> чтобы избежать дублирования
            # llama.cpp сам добавит этот токен
            content = messages[0].content
            if content.startswith("<|begin_of_text|>"):
                content = content[len("<|begin_of_text|>"):]
                logger.debug("🔧 Удален дублированный <|begin_of_text|> токен")
            
            return content

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
            logger.warning("⚠️ Пустой ответ от модели")
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

        # Проверяем, что контент не пустой
        if not content or not content.strip():
            logger.warning("⚠️ Модель вернула пустой контент, используем fallback")
            content = "Извините, не смог сгенерировать ответ. Попробуйте переформулировать запрос."

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

    def _is_service_request(self, query: str) -> bool:
        """Проверить, является ли запрос служебным (заголовки, теги, etc)."""
        query_lower = query.lower()

        # Ключевые слова для служебных запросов
        service_keywords = [
            # Генерация заголовков
            "generate a concise",
            "word title",
            "summarizing the chat",
            "generate title",
            "create title",
            "make title",
            "title for",
            "title with an emoji",
            "conversation title",

            # Генерация тегов
            "generate 1-3 broad tags",
            "categorizing the main themes",
            "tags categorizing",
            "broad tags",
            "specific subtopic tags",
            "themes of the chat",

            # Структурированные задачи Open WebUI
            "### task:",
            "### guidelines:",
            "### output:",
            "### examples:",
            "### chat history:",
            "<chat_history>",
            "</chat_history>",

            # Форматы ответов
            "json format:",
            "output format:",
            "response format:",
            "format: {",
            '"title":',
            '"tags":',

            # Анализ чата и метаданные
            "analyze the conversation",
            "summarize this chat",
            "extract key topics",
            "main subjects discussed",
            "primary language",
            "multilingual",

            # Системные инструкции
            "your entire response must consist",
            "without any introductory",
            "raw json object",
            "without markdown code fences",
            "parsing failure",
            "single, raw json",
            "ensure no conversational text",
        ]

        is_service = any(keyword in query_lower for keyword in service_keywords)

        if is_service:
            logger.info("🔍 Служебный запрос обнаружен", 
                       keywords=[kw for kw in service_keywords if kw in query_lower])

        return is_service

    def _is_title_generation_request(self, query: str) -> bool:
        """Проверить, является ли запрос генерацией заголовка."""
        # Теперь это частный случай служебного запроса
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
            "conversation title",
            "title with an emoji"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in title_keywords)

    async def _process_service_proxy_completion(self, request: ChatCompletionRequest, user_query: str, request_id: str) -> CompletionResponse:
        """Обработка служебного запроса как прямого proxy в LLM."""
        try:
            # Определяем специальные параметры для служебных запросов
            is_title = self._is_title_generation_request(user_query)
            
            # Форматируем служебный запрос в правильном формате Llama 3.1
            if is_title:
                # Для заголовков используем минимальный системный промпт
                formatted_prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "Generate a concise JSON title response.\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                max_tokens = 50
                temperature = 0.1
            else:
                # Для других служебных запросов используем базовый системный промпт
                formatted_prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "You are a helpful assistant. Provide concise, accurate responses.\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                max_tokens = 200
                temperature = 0.3
            
            # Оптимизированные параметры для быстрых служебных ответов
            response = self._llama.create_completion(
                prompt=formatted_prompt,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=20,  # Уменьшено для скорости
                top_p=0.8,  # Уменьшено для скорости
                repeat_penalty=1.0,  # Без штрафов для скорости
                # Критически важно для Llama 3.1 - stop токены
                stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"],
            )

            logger.info("✅ Служебный запрос обработан как прямой proxy в LLM", 
                       is_title=is_title, max_tokens=max_tokens, temp=temperature, request_id=request_id)

            return self._create_completion_response(response, request.model, "chat.completion")

        except Exception as e:
            logger.error("❌ Ошибка обработки служебного запроса", error=str(e), exc_info=True, request_id=request_id)
            return self._create_error_response(f"Ошибка обработки: {str(e)}")

    async def _process_service_proxy_completion_stream(self, request: ChatCompletionRequest, user_query: str) -> AsyncIterator[dict]:
        """Стриминг обработка служебного запроса как прямого proxy в LLM."""
        try:
            # Определяем специальные параметры для служебных запросов
            is_title = self._is_title_generation_request(user_query)
            
            # Форматируем служебный запрос в правильном формате Llama 3.1
            if is_title:
                # Для заголовков используем минимальный системный промпт
                formatted_prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "Generate a concise JSON title response.\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                max_tokens = 50
                temperature = 0.1
            else:
                # Для других служебных запросов используем базовый системный промпт
                formatted_prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "You are a helpful assistant. Provide concise, accurate responses.\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                max_tokens = 200
                temperature = 0.3
            
            # Оптимизированные параметры для быстрых служебных ответов
            stream = self._llama.create_completion(
                prompt=formatted_prompt,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=20,  # Уменьшено для скорости
                top_p=0.8,  # Уменьшено для скорости
                repeat_penalty=1.0,  # Без штрафов для скорости
                # Критически важно для Llama 3.1 - stop токены
                stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"],
            )

            logger.info("✅ Служебный запрос обработан как прямой proxy в LLM (streaming)", 
                       is_title=is_title, max_tokens=max_tokens, temp=temperature)

            # Стримим ответ напрямую от llama
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                logger.debug(f"🔄 Служебный chunk #{chunk_count}")
                
                yield {
                    "choices": [
                        {
                            "delta": {
                                "content": chunk.get("choices", [{}])[0].get("text", "")
                            },
                            "finish_reason": chunk.get("choices", [{}])[0].get("finish_reason")
                        }
                    ],
                    "object": "chat.completion.chunk",
                    "model": request.model
                }

        except Exception as e:
            logger.error("❌ Ошибка обработки служебного запроса", error=str(e), exc_info=True)
            yield self._create_stream_error(f"Ошибка обработки: {str(e)}")

    async def _process_full_rag_completion(self, request: ChatCompletionRequest, user_query: str, request_id: str) -> CompletionResponse:
        """Обработка пользовательского запроса с полным RAG."""
        try:
            # Создаем полный RAG промпт с поиском в базе знаний
            rag_prompt = await self._rag_service.create_rag_prompt(user_query)

            # 🔍 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ДЛЯ ДИАГНОСТИКИ
            logger.info("🔍 RAG промпт готов к отправке", 
                       rag_prompt_length=len(rag_prompt),
                       contains_begin_of_text="<|begin_of_text|>" in rag_prompt, request_id=request_id)
            
            # Выводим полный промпт для диагностики
            logger.debug("🔤 ПОЛНЫЙ RAG ПРОМПТ:", rag_prompt=rag_prompt)
            
            # Выводим первые 1000 символов в info логе
            logger.info("📝 RAG промпт (первые 1000 символов):", 
                       rag_preview=rag_prompt[:1000])

            # RAG промпт уже содержит системный контекст в правильном формате
            # Создаем простое пользовательское сообщение с RAG промптом
            rag_messages = [
                ChatMessage(role="user", content=rag_prompt)
            ]

            # Создаем RAG запрос с параметрами для пользовательских запросов
            llm_params = self._settings.merge_request_params(request, is_title=False)

            rag_request = ChatCompletionRequest(
                messages=rag_messages,
                model=request.model,
                stream=False,
                **llm_params
            )

            logger.info("✅ Полный RAG промпт создан",
                       original_len=len(user_query),
                       rag_len=len(rag_prompt), request_id=request_id)

            # Обрабатываем запрос
            return await self._process_completion(rag_request)

        except Exception as e:
            logger.error("❌ Ошибка полной RAG обработки", error=str(e), exc_info=True, request_id=request_id)
            return self._create_error_response(f"Ошибка обработки: {str(e)}")

    async def _process_full_rag_completion_stream(self, request: ChatCompletionRequest, user_query: str) -> AsyncIterator[dict]:
        """Стриминг обработка пользовательского запроса с полным RAG."""
        try:
            # Создаем полный RAG промпт с поиском в базе знаний
            rag_prompt = await self._rag_service.create_rag_prompt(user_query)

            # 🔍 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ДЛЯ ДИАГНОСТИКИ
            logger.info("🔍 RAG промпт готов к отправке (streaming)", 
                       rag_prompt_length=len(rag_prompt),
                       contains_begin_of_text="<|begin_of_text|>" in rag_prompt)
            
            # Выводим полный промпт для диагностики
            logger.debug("🔤 ПОЛНЫЙ RAG ПРОМПТ (streaming):", rag_prompt=rag_prompt)
            
            # Выводим первые 1000 символов в info логе
            logger.info("📝 RAG промпт (первые 1000 символов, streaming):", 
                       rag_preview=rag_prompt[:1000])

            # RAG промпт уже содержит системный контекст в правильном формате
            # Создаем простое пользовательское сообщение с RAG промптом
            rag_messages = [
                ChatMessage(role="user", content=rag_prompt)
            ]

            # Создаем RAG запрос с параметрами для пользователей
            llm_params = self._settings.merge_request_params(request, is_title=False)

            rag_request = ChatCompletionRequest(
                messages=rag_messages,
                model=request.model,
                stream=True,
                **llm_params
            )

            logger.info("✅ Полный RAG стриминг промпт создан")

            # Стримим ответ
            async for chunk in self._process_completion_stream(rag_request):
                yield chunk

        except Exception as e:
            logger.error("❌ Ошибка полного RAG стриминга", error=str(e), exc_info=True)
            yield self._create_stream_error(f"Ошибка обработки: {str(e)}")
