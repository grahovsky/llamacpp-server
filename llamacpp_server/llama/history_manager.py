"""Менеджер истории чата с ротацией токенов."""

from typing import Any

import structlog
from llama_cpp import Llama

logger = structlog.get_logger(__name__)


class ChatHistoryManager:
    """Управляет историей чата с ротацией при превышении лимита токенов."""

    def __init__(self, llama_model: Llama, max_tokens: int, reserve_tokens: int):
        """
        Args:
            llama_model: Экземпляр модели для подсчета токенов
            max_tokens: Максимальное количество токенов в истории
            reserve_tokens: Резерв токенов для нового ответа
        """
        self.llama_model = llama_model
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens

    def count_message_tokens(self, message: dict[str, Any]) -> int:
        """Подсчитывает токены в сообщении."""
        content = message.get("content", "")
        if not content:
            return 0

        try:
            # Добавляем служебные токены для роли
            role_prefix = f"<|{message.get('role', 'user')}|>\n"
            full_content = role_prefix + content

            tokens = self.llama_model.tokenize(full_content.encode())
            return len(tokens)
        except Exception as e:
            logger.warning("Ошибка подсчета токенов", error=str(e), content_length=len(content))
            # Примерная оценка: 1 токен = 4 символа
            return len(content) // 4 + 10  # +10 для служебных токенов

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Подсчитывает общее количество токенов в списке сообщений."""
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)
        return total

    def trim_history(self, messages: list[dict[str, Any]], new_message_tokens: int = 0) -> list[dict[str, Any]]:
        """
        Обрезает историю чата, оставляя место для нового сообщения.
        
        Args:
            messages: Список сообщений истории
            new_message_tokens: Количество токенов в новом сообщении
            
        Returns:
            Обрезанный список сообщений
        """
        if not messages:
            return messages

        # Максимальное количество токенов для истории с учетом нового сообщения
        max_history_tokens = self.max_tokens - self.reserve_tokens - new_message_tokens

        if max_history_tokens <= 0:
            logger.warning("Новое сообщение слишком большое",
                         new_tokens=new_message_tokens,
                         max_tokens=self.max_tokens)
            return []

        # Подсчитываем токены снизу вверх (сохраняем самые свежие сообщения)
        trimmed_messages = []
        current_tokens = 0

        # Всегда сохраняем системный промпт (первое сообщение с ролью system)
        system_message = None
        other_messages = messages.copy()

        if messages and messages[0].get("role") == "system":
            system_message = messages[0]
            system_tokens = self.count_message_tokens(system_message)
            current_tokens += system_tokens
            other_messages = messages[1:]

        # Добавляем сообщения с конца (самые свежие)
        for message in reversed(other_messages):
            message_tokens = self.count_message_tokens(message)

            if current_tokens + message_tokens > max_history_tokens:
                # Превысили лимит - останавливаемся
                break

            trimmed_messages.insert(0, message)
            current_tokens += message_tokens

        # Собираем результат: системное сообщение + обрезанная история
        result = []
        if system_message:
            result.append(system_message)
        result.extend(trimmed_messages)

        removed_count = len(messages) - len(result)
        if removed_count > 0:
            logger.info("Обрезана история чата",
                       removed_messages=removed_count,
                       total_tokens=current_tokens,
                       max_tokens=max_history_tokens)

        return result

    def prepare_messages_for_completion(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Подготавливает сообщения для отправки в модель с учетом лимитов токенов.
        
        Args:
            messages: Исходный список сообщений
            
        Returns:
            Обработанный список сообщений, готовый для модели
        """
        if not messages:
            return []

        # Получаем последнее сообщение пользователя
        last_message = messages[-1] if messages else {}
        last_message_tokens = self.count_message_tokens(last_message)

        # Обрезаем историю
        trimmed_messages = self.trim_history(messages[:-1], last_message_tokens)

        # Добавляем последнее сообщение
        if last_message:
            trimmed_messages.append(last_message)

        total_tokens = self.count_messages_tokens(trimmed_messages)
        logger.debug("Подготовлены сообщения для completion",
                    messages_count=len(trimmed_messages),
                    total_tokens=total_tokens,
                    max_context=self.max_tokens)

        return trimmed_messages
