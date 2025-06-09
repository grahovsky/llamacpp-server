"""Тесты для LlamaService."""

from unittest.mock import Mock

import pytest

from llamacpp_server.domain.models import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from llamacpp_server.llama.service import LlamaService


@pytest.fixture
def mock_llama():
    """Mock объект LLama модели."""
    return Mock()


@pytest.fixture
def llama_service(mock_llama):
    """Экземпляр LlamaService с mock моделью."""
    return LlamaService(mock_llama)


@pytest.mark.asyncio
async def test_chat_completion(llama_service, mock_llama):
    """Тест chat completion."""
    # Arrange
    mock_llama.create_completion.return_value = {
        "choices": [{
            "text": "Привет! Как дела?",
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    # Убираем create_chat_completion чтобы использовался fallback
    delattr(mock_llama, 'create_chat_completion')

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Привет!")]
    )

    # Act
    response = await llama_service.chat_completion(request)

    # Assert
    assert response.choices[0].text == "Привет! Как дела?"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 15
    mock_llama.create_completion.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completion_stream(llama_service, mock_llama):
    """Тест streaming chat completion."""
    # Arrange
    mock_stream = [
        {"choices": [{"delta": {"content": "При"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "вет!"}, "finish_reason": "stop"}]}
    ]
    mock_llama.create_completion.return_value = mock_stream

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Привет!")]
    )

    # Act
    chunks = []
    async for chunk in llama_service.chat_completion_stream(request):
        chunks.append(chunk)

    # Assert
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "При"
    assert chunks[1]["choices"][0]["delta"]["content"] == "вет!"


@pytest.mark.asyncio
async def test_text_completion(llama_service, mock_llama):
    """Тест text completion."""
    # Arrange
    mock_llama.create_completion.return_value = {
        "choices": [{
            "text": "Это тестовый ответ",
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9
        }
    }

    request = TextCompletionRequest(
        model="test-model",
        prompt="Напиши текст:"
    )

    # Act
    response = await llama_service.text_completion(request)

    # Assert
    assert response.choices[0].text == "Это тестовый ответ"
    assert response.usage.total_tokens == 9


@pytest.mark.asyncio
async def test_is_ready(llama_service):
    """Тест проверки готовности."""
    assert await llama_service.is_ready() is True


def test_format_chat_messages(llama_service):
    """Тест форматирования сообщений."""
    messages = [
        ChatMessage(role="system", content="Ты помощник"),
        ChatMessage(role="user", content="Привет"),
        ChatMessage(role="assistant", content="Привет!")
    ]

    formatted = llama_service._format_chat_messages(messages)

    expected = "System: Ты помощник\nUser: Привет\nAssistant: Привет!\nAssistant: "
    assert formatted == expected
