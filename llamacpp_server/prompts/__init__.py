"""Модуль для работы с промптами."""

from .service import PromptService
from .templates import get_prompt_template, PromptTemplateFactory

__all__ = ["PromptService", "get_prompt_template", "PromptTemplateFactory"]
