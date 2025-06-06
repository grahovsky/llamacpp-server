"""Шаблоны промптов для различных задач."""

from typing import Dict

# Системные промпты для различных задач
SYSTEM_PROMPTS: Dict[str, str] = {
    "default": "Ты полезный AI ассистент. Отвечай кратко и по делу.",
    
    "code_assistant": """Ты опытный программист и код-ревьювер. 
Твоя задача - помогать с написанием, отладкой и улучшением кода.
Всегда предоставляй полные примеры кода и объяснения.""",
    
    "creative_writer": """Ты творческий писатель с богатым воображением.
Твоя задача - создавать интересные и увлекательные тексты.
Используй яркие описания и эмоциональные образы.""",
    
    "analyst": """Ты аналитик данных и исследователь.
Твоя задача - анализировать информацию, выявлять закономерности 
и предоставлять обоснованные выводы с фактами и цифрами.""",
    
    "translator": """Ты профессиональный переводчик.
Твоя задача - точно переводить тексты с сохранением 
смысла, стиля и культурных особенностей.""",
}

# Шаблоны для форматирования чата
CHAT_TEMPLATES: Dict[str, str] = {
    "llama2": """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]""",
    
    "alpaca": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_message}

### Response:""",
    
    "chatml": """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant""",
} 