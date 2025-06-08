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
from ..retrieval.protocols import RAGServiceProtocol


logger = structlog.get_logger(__name__)


class LlamaService:
    """Сервис для работы с LLama моделью."""
    
    def __init__(
        self, 
        llama: Llama,
        rag_service: RAGServiceProtocol = None
    ) -> None:
        self._llama = llama
        self._rag_service = rag_service
        settings = get_settings()
        self._enable_rag = settings.enable_rag and rag_service is not None
        self._rag_search_k = settings.rag_search_k
        self._history_manager = ChatHistoryManager(
            llama_model=llama,
            max_tokens=settings.max_history_tokens,
            reserve_tokens=settings.context_reserve_tokens
        )
    
    @staticmethod
    def _is_technical_request(query: str) -> bool:
        """Определяет технические запросы от open-webui."""
        if not query:
            return False
        
        # Более строгие паттерны для технических запросов
        technical_patterns = [
            "### task:",
            "generate a concise",
            "generate 1-3 broad tags", 
            "summarizing the chat history",
            "categorizing the main themes",
            "json format:",
            '"title":', 
            '"tags":',
            "chat history:",
            "<chat_history>",
            "### guidelines:",
            "### output:",
            "3-5 word title",
            "broad tags categorizing",
            "main themes of the chat",
            "chat's primary language",
            "your entire response must consist solely",
            "raw json object",
        ]
        
        query_lower = query.lower()
        
        # Если запрос содержит технические паттерны
        has_technical_patterns = any(pattern in query_lower for pattern in technical_patterns)
        
        # Дополнительная проверка: если запрос начинается с ### - это технический
        starts_with_task = query_lower.strip().startswith("###")
        
        # Проверка на JSON структуру
        has_json_structure = '"title":' in query_lower or '"tags":' in query_lower
        
        return has_technical_patterns or starts_with_task or has_json_structure
        
    async def chat_completion(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Генерация chat completion с RAG поддержкой."""
        logger.info("Обработка chat completion запроса", 
                   model=request.model, 
                   rag_enabled=self._enable_rag)
        
        # Извлекаем последний пользовательский запрос для анализа
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        # Проверяем тип запроса
        is_technical = self._is_technical_request(user_query)
        
        # Логируем тип запроса
        if user_query:
            logger.info("🔍 ТИП ЗАПРОСА",
                       is_technical=is_technical,
                       query_preview=user_query[:100])

        # Добавляем системный промпт только для обычных пользовательских запросов
        has_system_message = any(msg.role == "system" for msg in request.messages)
        if not has_system_message and not is_technical:
            from ..prompts.templates import SYSTEM_PROMPTS
            system_message = ChatMessage(
                role="system", 
                content=SYSTEM_PROMPTS["rag_expert"]
            )
            # Добавляем системное сообщение в начало
            request.messages.insert(0, system_message)
            logger.info("✅ Добавлен системный промпт для RAG", 
                       prompt_type="rag_expert")
        
        # Если включен RAG и есть пользовательский запрос (не технический)
        if self._enable_rag and user_query and not is_technical:
            logger.info("🧠 RAG обработка запроса", 
                       query_preview=user_query[:100],
                       rag_service_available=self._rag_service is not None)
            
            try:
                # Ищем релевантный контекст
                context = await self._rag_service.search_relevant_context(
                    user_query, k=self._rag_search_k
                )
                
                if context:
                    logger.info("🎯 RAG контекст найден, улучшаем промпт", context_docs=len(context))
                    
                    # Улучшаем последнее сообщение пользователя
                    enhanced_query = await self._rag_service.enhance_prompt_with_context(
                        user_query, context
                    )
                    
                    # Создаем новый список сообщений с улучшенным запросом
                    enhanced_messages = []
                    for msg in request.messages:
                        if msg.role == "user" and msg.content == user_query:
                            # Заменяем последний пользовательский запрос
                            enhanced_messages.append(
                                ChatMessage(role="user", content=enhanced_query)
                            )
                        else:
                            enhanced_messages.append(msg)
                    
                    # Обновляем запрос
                    request = ChatCompletionRequest(
                        messages=enhanced_messages,
                        model=request.model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repeat_penalty=request.repeat_penalty,
                        seed=request.seed,
                        stream=request.stream
                    )
                    
                    logger.info("✅ RAG промпт обновлен", 
                               original_length=len(user_query),
                               enhanced_length=len(enhanced_query),
                               context_docs=len(context))
                else:
                    logger.warning("⚠️ RAG контекст не найден")
                    
            except Exception as e:
                logger.error("❌ Ошибка RAG обработки", error=str(e), exc_info=True)
        elif not self._enable_rag:
            logger.debug("RAG отключен в настройках")
        else:
            logger.debug("Нет пользовательского запроса для RAG")
        
        # Подготавливаем сообщения с учетом лимитов контекста
        messages_dict = [msg.dict() for msg in request.messages]
        
        # Логируем исходные сообщения для диагностики
        logger.info("🔍 ИСХОДНЫЕ СООБЩЕНИЯ ДЛЯ LLM",
                   messages_count=len(messages_dict),
                   messages_summary=[{
                       "role": msg.get("role", "unknown"),
                       "content_length": len(str(msg.get("content", "")))
                   } for msg in messages_dict])
        
        # Логируем полное содержимое последнего сообщения если это RAG (без технических)
        if messages_dict and self._enable_rag and not is_technical:
            last_msg = messages_dict[-1]
            logger.debug("🔍 ПОСЛЕДНЕЕ СООБЩЕНИЕ С RAG (ПОЛНОЕ)",
                       role=last_msg.get("role", "unknown"),
                       full_content=last_msg.get("content", ""))
        
        trimmed_messages = self._history_manager.prepare_messages_for_completion(messages_dict)
        
        # Логируем результат обрезки
        logger.info("🔍 СООБЩЕНИЯ ПОСЛЕ ОБРЕЗКИ",
                   original_count=len(messages_dict),
                   trimmed_count=len(trimmed_messages),
                   trimmed_summary=[{
                       "role": msg.get("role", "unknown"),
                       "content_length": len(str(msg.get("content", "")))
                   } for msg in trimmed_messages])
        
        # Форматируем сообщения в промпт
        formatted_prompt = self._format_chat_messages([
            ChatMessage(**msg) for msg in trimmed_messages
        ])
        
        # Логируем финальный промпт (только длину)
        logger.info("🔍 ФИНАЛЬНЫЙ ПРОМПТ ДЛЯ МОДЕЛИ",
                   prompt_length=len(formatted_prompt))
        
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
        """Стриминг chat completion с RAG поддержкой."""
        logger.info("Обработка streaming chat completion запроса", 
                   model=request.model,
                   rag_enabled=self._enable_rag)
        
        # Извлекаем последний пользовательский запрос для анализа
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        # Проверяем тип запроса для streaming
        is_technical = self._is_technical_request(user_query)
        
        # Логируем тип запроса
        if user_query:
            logger.info("🔍 STREAMING ТИП ЗАПРОСА",
                       is_technical=is_technical,
                       query_preview=user_query[:100])

        # Добавляем системный промпт только для обычных пользовательских запросов
        has_system_message = any(msg.role == "system" for msg in request.messages)
        if not has_system_message and not is_technical:
            from ..prompts.templates import SYSTEM_PROMPTS
            system_message = ChatMessage(
                role="system", 
                content=SYSTEM_PROMPTS["rag_expert"]
            )
            # Добавляем системное сообщение в начало
            request.messages.insert(0, system_message)
            logger.info("✅ Добавлен системный промпт для RAG streaming", 
                       prompt_type="rag_expert")
        
        # Если включен RAG и есть пользовательский запрос (не технический)
        if self._enable_rag and user_query and not is_technical:
            logger.info("🧠 RAG обработка streaming запроса")
            
            try:
                # Ищем релевантный контекст
                context = await self._rag_service.search_relevant_context(
                    user_query, k=self._rag_search_k
                )
                
                if context:
                    # Улучшаем последнее сообщение пользователя
                    enhanced_query = await self._rag_service.enhance_prompt_with_context(
                        user_query, context
                    )
                    
                    # Создаем новый список сообщений с улучшенным запросом
                    enhanced_messages = []
                    for msg in request.messages:
                        if msg.role == "user" and msg.content == user_query:
                            # Заменяем последний пользовательский запрос
                            enhanced_messages.append(
                                ChatMessage(role="user", content=enhanced_query)
                            )
                        else:
                            enhanced_messages.append(msg)
                    
                    # Обновляем запрос
                    request = ChatCompletionRequest(
                        messages=enhanced_messages,
                        model=request.model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repeat_penalty=request.repeat_penalty,
                        seed=request.seed,
                        stream=request.stream
                    )
                    
                    logger.info("✅ RAG промпт обновлен для streaming", context_docs=len(context))
                    
            except Exception as e:
                logger.error("❌ Ошибка RAG обработки в streaming", error=str(e))
        
        # Подготавливаем сообщения с учетом лимитов контекста
        messages_dict = [msg.dict() for msg in request.messages]
        
        # Логируем для streaming тоже (но короче)
        logger.info("🔍 STREAMING: исходные сообщения",
                   messages_count=len(messages_dict))
        
        if messages_dict and self._enable_rag:
            last_msg = messages_dict[-1]
            logger.info("🔍 STREAMING: RAG сообщение длина",
                       content_length=len(str(last_msg.get("content", ""))))
        
        trimmed_messages = self._history_manager.prepare_messages_for_completion(messages_dict)
        
        logger.info("🔍 STREAMING: после обрезки",
                   original_count=len(messages_dict),
                   trimmed_count=len(trimmed_messages))
        
        formatted_prompt = self._format_chat_messages([
            ChatMessage(**msg) for msg in trimmed_messages
        ])
        
        logger.info("🔍 STREAMING: финальный промпт длина",
                   prompt_length=len(formatted_prompt))
        
        # Логируем сам промпт для debug
        logger.info("🔍 STREAMING: промпт для модели",
                   prompt_preview=formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)
        
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
        """Форматирование сообщений в промпт для Mistral-7B."""
        # Убираем <s> - llama.cpp добавляет автоматически
        formatted = ""
        system_content = ""
        user_content = ""
        
        # Собираем системный промпт и пользовательский контент
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "user":
                user_content = msg.content
            elif msg.role == "assistant":
                # Для истории диалога (пока не используем)
                pass
        
        # Формат Mistral без <s> (llama.cpp добавит сам)
        if system_content and user_content:
            formatted = f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"
        elif user_content:
            formatted = f"[INST] {user_content} [/INST]"
        
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