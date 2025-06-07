# 🔧 Конфигурация

Полное руководство по настройке llamacpp-server.

## Переменные окружения

Все настройки конфигурируются через переменные окружения с префиксом `LLAMACPP_`.

### Основные настройки

```bash
# === Модель ===
LLAMACPP_MODEL_PATH=models/model.gguf  # Путь к GGUF модели
LLAMACPP_DEV_MODE=false                # Режим разработки (mock модель)

# === Сервер ===
LLAMACPP_HOST=0.0.0.0                  # IP адрес (0.0.0.0 для всех интерфейсов)
LLAMACPP_PORT=8090                     # Порт сервера

# === Логирование ===
LLAMACPP_LOG_LEVEL=DEBUG               # DEBUG, INFO, WARNING, ERROR
LLAMACPP_VERBOSE=false                 # Подробный вывод llama.cpp
```

### Параметры модели

```bash
# === Контекстное окно ===
LLAMACPP_N_CTX=4096                    # Размер контекста (токены)
LLAMACPP_N_BATCH=512                   # Размер батча для обработки
LLAMACPP_N_THREADS=8                   # Количество CPU потоков

# === Управление историей ===
LLAMACPP_MAX_HISTORY_TOKENS=3000       # Максимум токенов в истории чата
LLAMACPP_CONTEXT_RESERVE_TOKENS=1000   # Резерв токенов для ответа

# === Память ===
LLAMACPP_USE_MMAP=true                 # Memory mapping для модели
LLAMACPP_USE_MLOCK=false               # Memory locking
```

### GPU параметры

```bash
# === CUDA ===
LLAMACPP_N_GPU_LAYERS=0                # Количество слоев на GPU (0=CPU, -1=все)
LLAMACPP_MAIN_GPU=0                    # ID основной GPU
LLAMACPP_TENSOR_SPLIT="0.7,0.3"       # Разделение между GPU (через запятую)
```

### Параметры генерации

```bash
# === Генерация ===
LLAMACPP_TEMPERATURE=0.7               # Температура (0.0-2.0)
LLAMACPP_CHAT_FORMAT=llama-2           # Формат чата (llama-2, chatml, etc.)
```

## Файл .env

Вместо переменных окружения можно использовать файл `.env` в корне проекта:

```bash
# .env файл
LLAMACPP_MODEL_PATH=models/mistral-7b.gguf
LLAMACPP_N_GPU_LAYERS=20
LLAMACPP_N_CTX=4096
LLAMACPP_PORT=8090
```

## Конфигурация через код

Настройки также можно изменить программно:

```python
from llamacpp_server.config.settings import get_settings

settings = get_settings()
settings.n_gpu_layers = 20
settings.temperature = 0.8
```

## Примеры конфигураций

### Высокая производительность (GPU)
```bash
LLAMACPP_N_GPU_LAYERS=-1              # Вся модель на GPU
LLAMACPP_N_CTX=8192                   # Большой контекст
LLAMACPP_N_BATCH=1024                 # Большой батч
LLAMACPP_USE_MMAP=true
LLAMACPP_USE_MLOCK=true
```

### Экономия памяти (CPU)
```bash
LLAMACPP_N_GPU_LAYERS=0               # Только CPU
LLAMACPP_N_CTX=2048                   # Меньший контекст
LLAMACPP_N_BATCH=256                  # Меньший батч
LLAMACPP_N_THREADS=4                  # Меньше потоков
```

### Режим разработки
```bash
LLAMACPP_DEV_MODE=true                # Mock модель
LLAMACPP_LOG_LEVEL=DEBUG              # Подробные логи
LLAMACPP_VERBOSE=true                 # Дополнительная информация
```

### Множественные GPU
```bash
LLAMACPP_N_GPU_LAYERS=30              # Частично на GPU
LLAMACPP_MAIN_GPU=0                   # Основная GPU
LLAMACPP_TENSOR_SPLIT="0.6,0.4"      # 60% на GPU 0, 40% на GPU 1
```

## Производительность

### Факторы влияющие на скорость:

1. **GPU vs CPU**: GPU в 5-10 раз быстрее
2. **Размер контекста**: Больше контекст = медленнее
3. **Количество слоев на GPU**: Больше слоев = быстрее
4. **Размер модели**: Меньше модель = быстрее

### Оптимизация:

```bash
# Для максимальной скорости
LLAMACPP_N_GPU_LAYERS=-1              # Вся модель на GPU
LLAMACPP_N_BATCH=1024                 # Большой батч
LLAMACPP_USE_MMAP=true                # Memory mapping

# Для экономии VRAM
LLAMACPP_N_GPU_LAYERS=20              # Часть слоев на GPU
LLAMACPP_N_CTX=2048                   # Меньший контекст
```

## Мониторинг

### Логи
```bash
# Изменить уровень логирования
export LLAMACPP_LOG_LEVEL=INFO

# Включить подробные логи llama.cpp
export LLAMACPP_VERBOSE=true
```

### Диагностика GPU
```bash
# Проверка GPU
nvidia-smi

# Проверка поддержки CUDA в коде
uv run python -c "import llama_cpp; from llama_cpp import llama_cpp; print('CUDA:', llama_cpp.llama_supports_gpu_offload())"
```

## Устранение неполадок

### Проблемы с памятью
- Уменьшите `LLAMACPP_N_CTX`
- Уменьшите `LLAMACPP_N_GPU_LAYERS`
- Установите `LLAMACPP_USE_MMAP=false`

### Медленная генерация
- Увеличьте `LLAMACPP_N_GPU_LAYERS`
- Увеличьте `LLAMACPP_N_BATCH`
- Проверьте загрузку GPU с `nvidia-smi`

### Ошибки контекста
- Уменьшите `LLAMACPP_MAX_HISTORY_TOKENS`
- Увеличьте `LLAMACPP_CONTEXT_RESERVE_TOKENS` 