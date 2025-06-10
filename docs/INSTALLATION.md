# 🏗️ Установка и настройка

Пошаговое руководство по установке llamacpp-server.

## Требования

- **Python**: 3.11 или выше
- **Система**: Linux, macOS, Windows
- **Память**: минимум 8GB RAM для моделей 7B
- **Диск**: 5-20GB для моделей в зависимости от размера

## Установка

### 1. Клонирование репозитория

```bash
git clone <repo-url>
cd llamacpp-server
```

### 2. Установка зависимостей

#### Базовая установка (CPU)
```bash
# Установка только CPU поддержки
uv sync --extra cpu
# Для GPU
uv sync --extra gpu
```

#### Установка для разработки
```bash
# Все зависимости включая dev tools
uv sync --extra all
```

#### GPU поддержка
См. [GPU Setup Guide](GPU_SETUP.md) для детальных инструкций.

### 3. Конфигурация

Создайте файл `.env` в корне проекта:

```bash
# === Основные настройки ===
LLAMACPP_MODEL_PATH=models/model.gguf
LLAMACPP_DEV_MODE=false

# === Сервер ===
LLAMACPP_HOST=0.0.0.0
LLAMACPP_PORT=8090

# === Контекст ===
LLAMACPP_N_CTX=4096
LLAMACPP_MAX_HISTORY_TOKENS=3000
LLAMACPP_CONTEXT_RESERVE_TOKENS=1000

# === GPU (опционально) ===
LLAMACPP_N_GPU_LAYERS=0  # 0=CPU, >0=GPU слои
LLAMACPP_MAIN_GPU=0
```

### 4. Загрузка модели

```bash
# Создайте папку для моделей
mkdir -p models

# Загрузите GGUF модель (пример)
wget -O models/model.gguf https://example.com/model.gguf
```

### 5. Первый запуск

#### Режим разработки (без модели)
```bash
LLAMACPP_DEV_MODE=true uv run python -m llamacpp_server.main
```

#### С реальной моделью
```bash
uv run python -m llamacpp_server.main
```

### 6. Проверка работы

```bash
# Проверка здоровья
curl http://localhost:8090/health

# Список моделей
curl http://localhost:8090/api/tags

# Тестовый запрос
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Устранение неполадок

### Проблема: "Модель не найдена"
```bash
# Проверьте путь к модели
ls -la models/

# Убедитесь что путь правильный в .env
echo $LLAMACPP_MODEL_PATH
```

### Проблема: "Не хватает памяти"
```bash
# Уменьшите контекст
export LLAMACPP_N_CTX=2048

# Или используйте меньшую модель
```

### Проблема: "Порт занят"
```bash
# Проверьте что порт свободен
netstat -tlnp | grep 8090

# Или измените порт
export LLAMACPP_PORT=8091
```

## Следующие шаги

- [GPU поддержка](GPU_SETUP.md) для ускорения
- [Конфигурация](CONFIGURATION.md) для тонкой настройки
- [API документация](API.md) для интеграции 