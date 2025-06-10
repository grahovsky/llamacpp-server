#!/usr/bin/env python3

import asyncio
from llamacpp_server.prompts.service import PromptService

async def test_prompt_structure():
    print("🧪 Тестирование новой структуры промпта...")
    
    ps = PromptService()
    
    # Тестируем создание RAG промпта
    query = "как перейти с okd на deckhouse?"
    context = [
        "Для перехода с OKD на Deckhouse необходимо:",
        "1. Создать обращение в ЕСО",
        "2. Подготовить миграцию приложений"
    ]
    
    prompt = await ps.create_rag_prompt(query, context)
    
    print("✅ Структура промпта:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    
    print("\n🔍 Проверки:")
    print(f"• Содержит токены Llama 3.1: {'✅' if '<|begin_of_text|>' in prompt else '❌'}")
    print(f"• Содержит system роль: {'✅' if '<|start_header_id|>system<|end_header_id|>' in prompt else '❌'}")
    print(f"• Содержит user роль: {'✅' if '<|start_header_id|>user<|end_header_id|>' in prompt else '❌'}")
    print(f"• Содержит assistant роль: {'✅' if '<|start_header_id|>assistant<|end_header_id|>' in prompt else '❌'}")
    expected_ending = 'assistant<|end_header_id|>\n\n'
    print(f"• Заканчивается правильно: {'✅' if prompt.endswith(expected_ending) else '❌'}")
    
    print(f"\n📏 Размер промпта: {len(prompt)} символов")

if __name__ == "__main__":
    asyncio.run(test_prompt_structure()) 