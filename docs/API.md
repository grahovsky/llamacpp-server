# 📡 API Документация

llamacpp-server предоставляет множественные API интерфейсы для совместимости с различными клиентами.

## Базовые эндпоинты

### Health Check
```bash
GET /health
# Ответ: {"status": "healthy", "model_ready": true}
```

## OpenAI API

Совместимость с OpenAI API для интеграции с существующими клиентами.

### Chat Completions
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama",
  "messages": [
    {"role": "user", "content": "Привет!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 100
}
```

#### Streaming
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama", 
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}
```

### Text Completions
```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "llama",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.7
}
```

### Models List
```bash
GET /v1/models
# Ответ: список доступных моделей
```

## Ollama API

Полная совместимость с Ollama API.

### Chat
```bash
POST /api/chat
Content-Type: application/json

{
  "model": "llama",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false
}
```

### Generate
```bash
POST /api/generate
Content-Type: application/json

{
  "model": "llama",
  "prompt": "Why is the sky blue?",
  "stream": false
}
```

### Model Management
```bash
# Список моделей
GET /api/tags

# Информация о модели
POST /api/show
{"name": "llama"}

# Статус
GET /api/ps

# Версия
GET /api/version
```

## Open WebUI API

Специальные эндпоинты для Open WebUI.

### Chat
```bash
POST /api/chat
Content-Type: application/json

{
  "model": "llama",
  "messages": [{"role": "user", "content": "Hi"}],
  "stream": true
}
```

### Generate
```bash
POST /api/generate
Content-Type: application/json

{
  "model": "llama", 
  "prompt": "Hello world",
  "stream": true
}
```

## Параметры запросов

### Общие параметры
- `model`: Имя модели (обычно "llama" или полное имя)
- `stream`: true/false - стриминг ответа
- `temperature`: 0.0-2.0 - креативность (по умолчанию 0.7)
- `max_tokens`: Максимум токенов в ответе
- `top_p`: 0.0-1.0 - nucleus sampling
- `top_k`: >0 - top-k sampling
- `repeat_penalty`: >0 - штраф за повторы
- `seed`: Число для воспроизводимости

### Параметры чата
- `messages`: Массив сообщений с `role` и `content`
- Роли: `system`, `user`, `assistant`

## Примеры использования

### Python с OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="dummy"  # Не используется, но требуется
)

response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### curl с стримингом
```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }' \
  --no-buffer
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8090/v1/chat/completions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: 'llama',
    messages: [{role: 'user', content: 'Hello!'}]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## Коды ответов

- `200` - Успешный запрос
- `400` - Неверный запрос (проверьте JSON)
- `404` - Эндпоинт не найден
- `422` - Ошибка валидации
- `500` - Внутренняя ошибка сервера

## Ограничения

- **Контекст**: максимум 4096 токенов
- **История**: автоматическая ротация при превышении лимита
- **Модели**: одновременно загружена только одна модель
- **Concurrency**: несколько запросов обрабатываются последовательно

## WebUI интеграция

### Open WebUI
```bash
# В настройках Open WebUI добавьте:
# API Base URL: http://localhost:8090
# Модель будет автоматически обнаружена
```

### Другие клиенты
Любой клиент совместимый с OpenAI API или Ollama API будет работать. 