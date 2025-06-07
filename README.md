# 🦙 llamacpp-server

Современный асинхронный сервер для llama.cpp с GPU поддержкой, совместимый с OpenAI API, Ollama и Open WebUI.

## ✨ Возможности

- 🚀 **GPU ускорение** с CUDA поддержкой
- 🔄 **Автоматическая ротация контекста** - никаких ошибок переполнения
- 🌐 **Множественная API совместимость**:
  - OpenAI API (`/v1/chat/completions`, `/v1/completions`)
  - Ollama API (`/api/chat`, `/api/generate`, `/api/tags`)
  - Open WebUI API (`/api/*`)
- 📡 **Стриминг** - токен за токеном в реальном времени
- 🧠 **Умное управление историей** чата
- 🐍 **Python 2025 стандарты** - uv, ruff, mypy, pytest
- 🏗️ **Clean Architecture** - слоистая архитектура

## 🚀 Быстрый старт

### Установка зависимостей
```bash
# Клонируем репозиторий
git clone <repo-url>
cd llamacpp-server

# Устанавливаем зависимости
uv sync --extra cpu

# Для GPU поддержки - см. docs/GPU_SETUP.md
```

### Запуск сервера
```bash
# Режим разработки (без модели)
LLAMACPP_DEV_MODE=true uv run python -m llamacpp_server.main

# С реальной моделью
LLAMACPP_MODEL_PATH=models/model.gguf uv run python -m llamacpp_server.main

# С GPU ускорением
LLAMACPP_N_GPU_LAYERS=20 uv run python -m llamacpp_server.main
```

### Проверка работы
```bash
# Список моделей
curl http://localhost:8090/api/tags

# Chat completion
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Привет!"}]}'
```

## 📚 Документация

- [🏗️ Установка и настройка](docs/INSTALLATION.md)
- [🎮 GPU поддержка](docs/GPU_SETUP.md)
- [🔧 Конфигурация](docs/CONFIGURATION.md)
- [📡 API документация](docs/API.md)
- [🧪 Тестирование](docs/TESTING.md)
- [📋 Changelog](CHANGELOG.md)

## ⚙️ Конфигурация

Основные переменные окружения:

```bash
# === Модель ===
LLAMACPP_MODEL_PATH=models/model.gguf
LLAMACPP_DEV_MODE=false

# === GPU ===
LLAMACPP_N_GPU_LAYERS=20    # 0=CPU, -1=все слои на GPU
LLAMACPP_MAIN_GPU=0
LLAMACPP_TENSOR_SPLIT="0.7,0.3"

# === Контекст ===
LLAMACPP_N_CTX=4096
LLAMACPP_MAX_HISTORY_TOKENS=3000
LLAMACPP_CONTEXT_RESERVE_TOKENS=1000

# === Сервер ===
LLAMACPP_HOST=0.0.0.0
LLAMACPP_PORT=8090
```

## 🏗️ Архитектура

```
llamacpp_server/
├── config/          # Настройки и конфигурация
├── domain/          # Модели данных и интерфейсы
├── infrastructure/  # Внешние зависимости (DI)
├── llama/          # Логика работы с llama.cpp
├── prompts/        # Системные промпты
└── web/            # FastAPI роутеры и API
```

## 🧪 Тестирование

```bash
# Запуск всех тестов
uv run pytest

# С покрытием кода
uv run pytest --cov=llamacpp_server

# Только определенные тесты
uv run pytest tests/test_config.py
```

## 🔧 Разработка

```bash
# Установка с dev зависимостями
uv sync --extra all

# Линтинг и форматирование
uv run ruff check .
uv run ruff format .

# Проверка типов
uv run mypy llamacpp_server/
```

## 📊 Производительность

| Конфигурация | Скорость генерации |
|--------------|-------------------|
| CPU только | ~2-5 токенов/сек |
| RTX 3060 | ~15-25 токенов/сек |
| RTX 4090 | ~50-80 токенов/сек |

## 🤝 Совместимость

- **Python**: 3.11+
- **Операционные системы**: Linux, macOS, Windows
- **GPU**: NVIDIA CUDA 12.1+, AMD ROCm, Apple Metal
- **API клиенты**: OpenAI SDK, Ollama, Open WebUI, LangChain

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 🙏 Благодарности

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - основная библиотека
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python биндинги
- [FastAPI](https://fastapi.tiangolo.com/) - веб фреймворк 