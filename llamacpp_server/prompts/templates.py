"""Шаблоны промптов для разных типов моделей."""

from abc import ABC, abstractmethod
from typing import Protocol


class PromptTemplateProtocol(Protocol):
    """Протокол для шаблонов промптов."""
    
    def format_rag_prompt(self, query: str, context: list[str], system_message: str = "") -> str:
        """Форматировать RAG промпт."""
        ...
    
    def format_chat_prompt(self, messages: list[dict], system_message: str = "") -> str:
        """Форматировать чат промпт."""
        ...


class InstructTemplate:
    """Шаблон для instruct моделей (Llama 3.1 Instruct, Alpaca, Vicuna)."""
    
    def format_rag_prompt(self, query: str, context: list[str], system_message: str = "") -> str:
        """
        Форматировать RAG промпт для instruct модели.
        
        Instruct модели работают с простыми промптами без сложной структуры ролей.
        """
        context_text = "\n\n".join(context) if context else ""
        
        if system_message:
            base_instruction = system_message
        else:
            base_instruction = (
                "Ты — строгий ассистент по документации.\n\n"
                "КРИТИЧЕСКИ ВАЖНО:\n"
                "• Отвечай ТОЛЬКО на основе предоставленного контекста\n"
                "• ОБЯЗАТЕЛЬНО включай ссылки на источники в формате: [Название статьи](URL)\n"
                "• В конце ответа добавляй раздел \"Источники:\" со всеми ссылками\n"
                "• Если информации нет в контексте - прямо так и скажи\n"
                "• НЕ используй внешние знания\n\n"
                "ПРИМЕР ФОРМАТА ОТВЕТА:\n"
                "Ответ на вопрос...\n\n"
                "Источники:\n"
                "• [Название статьи 1](URL1)\n"
                "• [Название статьи 2](URL2)"
            )
        
        if context_text:
            prompt = f"""{base_instruction}

Контекст:
{context_text}

Вопрос: {query}

Ответ:"""
        else:
            prompt = f"""{base_instruction}

Вопрос: {query}

Ответ:"""
        
        return prompt
    
    def format_chat_prompt(self, messages: list[dict], system_message: str = "") -> str:
        """Форматировать чат промпт для instruct модели."""
        # Instruct модели обычно работают в формате "Instruction: ... Response:"
        formatted_parts = []
        
        if system_message:
            formatted_parts.append(f"Инструкция: {system_message}")
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_parts.append(f"Пользователь: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Ассистент: {content}")
        
        formatted_parts.append("Ассистент:")
        return "\n\n".join(formatted_parts)


class ChatTemplate:
    """Шаблон для chat моделей (Llama 3.1 Chat, ChatGLM)."""
    
    def format_rag_prompt(self, query: str, context: list[str], system_message: str = "") -> str:
        """
        Форматировать RAG промпт для chat модели.
        
        Chat модели используют структуру с ролями и специальными токенами.
        """
        context_text = "\n\n".join(context) if context else ""
        
        if system_message:
            base_instruction = system_message
        else:
            base_instruction = (
                "Отвечай на вопросы на основе предоставленного контекста. "
                "Обязательно добавляй ссылки в формате [Название](URL)."
            )
        
        # Llama 3.1 Chat формат - СТРОГО по спецификации
        if context_text:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{base_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

Контекст:
{context_text}

Вопрос: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{base_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def format_chat_prompt(self, messages: list[dict], system_message: str = "") -> str:
        """Форматировать чат промпт для chat модели."""
        formatted_parts = ["<|begin_of_text|>"]
        
        # Добавляем system message если есть
        if system_message:
            formatted_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>")
        
        # Добавляем сообщения
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant":
                formatted_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        # Добавляем начало ответа ассистента
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(formatted_parts)


class PromptTemplateFactory:
    """Фабрика для создания шаблонов промптов."""
    
    @staticmethod
    def get_template(model_type: str) -> PromptTemplateProtocol:
        """Получить шаблон для типа модели."""
        if model_type == "instruct":
            return InstructTemplate()
        elif model_type == "chat":
            return ChatTemplate()
        else:
            # По умолчанию используем instruct как более универсальный
            return InstructTemplate()


def get_prompt_template(model_type: str) -> PromptTemplateProtocol:
    """Получить шаблон промпта для типа модели."""
    return PromptTemplateFactory.get_template(model_type)
