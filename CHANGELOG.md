# Changelog

Все значимые изменения в проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
проект следует [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 🔄 Автоматическая ротация контекстного окна для длинных историй чата
- 🧠 ChatHistoryManager для умного управления токенами
- 🎮 Полная поддержка GPU через CUDA Toolkit
- 📊 Диагностика GPU устройств при запуске
- 🔧 Настройки управления контекстом через переменные окружения
- 📏 Подсчет реальных токенов через llama-cpp-python
- 🛡️ Сохранение системных сообщений при ротации
- 📝 Подробная документация по GPU установке

### Changed
- ⚡ Увеличен размер контекста с 2048 до 4096 токенов
- 🏗️ Интеграция ChatHistoryManager в LlamaService
- 🎭 Улучшена MockLlama с поддержкой tokenize()
- 📦 Обновлена структура pyproject.toml для GPU зависимостей
- 🔄 Улучшен процесс установки GPU поддержки

### Fixed
- 🚫 Устранена ошибка "Requested tokens exceed context window"
- 💾 Исправлена память при переполнении контекста
- 🎯 Гарантированный резерв токенов для ответа модели

### Technical Details
- **Контекстное окно**: 4096 токенов (было 2048)
- **Лимит истории**: 3000 токенов
- **Резерв для ответа**: 1000 токенов
- **Тестирование**: История 12,367 → 24 токена при ротации
- **GPU поддержка**: CUDA 12.8 на RTX 3070

---

## [0.1.0] - 2025-06-07

### Added
- 🚀 Первоначальная версия сервера llama.cpp
- 🌐 Поддержка множественных API:
  - OpenAI API (`/v1/chat/completions`, `/v1/completions`)
  - Ollama API (`/api/chat`, `/api/generate`, `/api/tags`)
  - Open WebUI API (`/api/*`)
- 📡 Стриминг генерации токен за токеном
- 🏗️ Clean Architecture с слоистой структурой
- 🐍 Python 2025 стандарты (uv, ruff, mypy, pytest)
- 🔧 Pydantic Settings v2 для конфигурации
- 📊 structlog для логирования
- 🧪 Comprehensive test suite
- 🔄 dependency-injector для DI
- 🛡️ SOLID принципы во всем коде
- 🎭 MockLlama для режима разработки

### Technical Stack
- **Python**: 3.11+
- **Framework**: FastAPI + uvicorn
- **LLM**: llama.cpp + llama-cpp-python
- **Architecture**: Clean Architecture
- **Config**: Pydantic Settings v2
- **Logging**: structlog
- **DI**: dependency-injector
- **Testing**: pytest + httpx
- **Code Quality**: ruff + mypy
- **Package Manager**: uv

[Unreleased]: https://github.com/user/llamacpp-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/user/llamacpp-server/releases/tag/v0.1.0 