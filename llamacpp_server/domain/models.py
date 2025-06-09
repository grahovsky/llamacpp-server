"""Модели данных приложения."""


from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Сообщение в чате."""

    role: str = Field(..., description="Роль: system, user, assistant")
    content: str = Field(..., description="Содержимое сообщения")


class CompletionRequest(BaseModel):
    """Запрос на генерацию текста."""

    # Общие поля
    model: str = Field("llama-cpp", description="Имя модели")
    max_tokens: int | None = Field(2048, ge=1, description="Максимум токенов")
    temperature: float | None = Field(0.2, ge=0.0, le=2.0, description="Температура")
    top_p: float | None = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int | None = Field(40, ge=1, description="Top-k sampling")
    repeat_penalty: float | None = Field(1.1, ge=0.0, description="Штраф за повторы")
    seed: int | None = Field(None, description="Seed для воспроизводимости")
    stream: bool = Field(False, description="Потоковый режим")


class ChatCompletionRequest(CompletionRequest):
    """Запрос на chat completion."""

    messages: list[ChatMessage] = Field(..., description="Список сообщений")


class TextCompletionRequest(CompletionRequest):
    """Запрос на text completion."""

    prompt: str = Field(..., description="Текст промпта")


class Choice(BaseModel):
    """Вариант ответа."""

    index: int
    message: ChatMessage | None = None
    text: str | None = None
    finish_reason: str


class Usage(BaseModel):
    """Информация об использовании токенов."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Ответ на запрос completion."""

    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
