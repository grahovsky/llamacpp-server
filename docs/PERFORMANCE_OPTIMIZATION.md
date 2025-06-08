# Оптимизация производительности LLM

## 📊 Анализ текущих настроек

### ✅ Хорошие настройки:
- `n_gpu_layers: 30` - использование GPU
- `use_mmap: True` - эффективная загрузка модели
- `n_batch: 512` - разумный размер батча
- `Q4_K_M` квантизация - баланс скорости и качества

### ⚡ Возможности оптимизации:

## 1. GPU Оптимизация

### Увеличение слоев на GPU
```bash
# Для RTX 3070 (8GB VRAM) можно увеличить
LLAMACPP_N_GPU_LAYERS=35  # Попробуйте постепенно увеличивать до 40+
```

### Flash Attention (если поддерживается)
```bash
LLAMACPP_FLASH_ATTN=true
```

### Оптимизация батча для GPU
```bash
LLAMACPP_N_BATCH=1024     # Увеличить для GPU
LLAMACPP_N_UBATCH=512     # Микро-батч для CUDA
```

## 2. Контекст и Память

### Оптимальный размер контекста
```bash
# Уменьшить для ускорения (текущий: 8192)
LLAMACPP_N_CTX=4096       # Для большинства задач достаточно

# Или увеличить если нужен большой контекст
LLAMACPP_N_CTX=16384      # Для сложных RAG задач
```

### KV-cache оптимизация
```bash
LLAMACPP_N_CTX_TRAIN=131072   # Размер контекста при обучении модели
LLAMACPP_ROPE_FREQ_BASE=10000  # RoPE frequency base
```

## 3. CPU Оптимизация

### Максимальное использование CPU
```python
import os
n_threads = os.cpu_count()  # Все доступные потоки
```

### NUMA оптимизация (для серверов)
```bash
numactl --cpunodebind=0 --membind=0 python -m llamacpp_server.main
```

## 4. Квантизация моделей

### Доступные форматы (от быстрых к качественным):
- `Q2_K` - самая быстрая, низкое качество
- `Q3_K_M` - быстрая, приемлемое качество  
- `Q4_0` - стандартная, хорошая скорость
- `Q4_K_M` - **текущая**, баланс скорости/качества ✅
- `Q5_K_M` - высокое качество, медленнее
- `Q6_K` - максимальное качество, самая медленная

### Рекомендации по выбору:
- **Продакшн**: `Q4_K_M` или `Q4_K_S`
- **Разработка**: `Q3_K_M` для скорости
- **Высокое качество**: `Q5_K_M`

## 5. RAG Оптимизация

### Уменьшение количества документов
```bash
LLAMACPP_RAG_SEARCH_K=5   # Вместо 8, ускорит поиск
LLAMACPP_MAX_CONTEXT_LENGTH=4000  # Уменьшить RAG контекст
```

### Кэширование эмбеддингов
```python
# В настройках - использовать FAISS с кэшированием
enable_embedding_cache: bool = True
```

### Параллельная обработка RAG
```python
# Асинхронная обработка эмбеддингов
concurrent_rag_queries: int = 2
```

## 6. Настройки генерации

### Ускорение генерации
```bash
LLAMACPP_TEMPERATURE=0.1      # Низкая температура = меньше вариативности
LLAMACPP_TOP_P=0.9           # Nucleus sampling
LLAMACPP_TOP_K=40            # Ограничение токенов
LLAMACPP_REPEAT_PENALTY=1.1   # Избежание повторов
```

### Ограничение длины ответов
```python
default_max_tokens: int = 512  # Вместо неограниченного
```

## 7. Системные оптимизации

### Приоритет процесса
```bash
# Linux - высокий приоритет
nice -n -10 python -m llamacpp_server.main

# Или через systemd
[Service]
Nice=-10
IOSchedulingClass=1
IOSchedulingPriority=4
```

### Управление памятью
```bash
# Использование huge pages (Linux)
echo 2048 > /proc/sys/vm/nr_hugepages

# Настройка swappiness
echo 10 > /proc/sys/vm/swappiness
```

### SSD оптимизация
```bash
# Размещение модели на NVMe SSD
LLAMACPP_MODEL_PATH=/fast_ssd/models/llama-3.1-8b.gguf
```

## 8. Мониторинг производительности

### Метрики для отслеживания:
- **Tokens/sec** - основная метрика скорости
- **TTFT** (Time To First Token) - время до первого токена
- **GPU utilization** - использование GPU
- **Memory usage** - потребление памяти
- **Queue time** - время ожидания в очереди

### Инструменты мониторинга:
```bash
# GPU мониторинг
nvidia-smi -l 1

# CPU/Memory
htop

# Сетевые метрики
iftop
```

## 9. Профилирование

### Профилирование с cProfile
```bash
python -m cProfile -o profile.stats -m llamacpp_server.main
```

### Анализ bottlenecks
```python
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative').print_stats(20)
```

## 🚀 Готовые профили оптимизации

### Максимальная скорость (.env.speed)
```bash
LLAMACPP_N_CTX=2048
LLAMACPP_N_BATCH=1024
LLAMACPP_N_GPU_LAYERS=40
LLAMACPP_TEMPERATURE=0.1
LLAMACPP_RAG_SEARCH_K=3
LLAMACPP_MAX_CONTEXT_LENGTH=2000
LLAMACPP_LOG_LEVEL=INFO
```

### Баланс скорость/качество (.env.balanced)
```bash
LLAMACPP_N_CTX=4096
LLAMACPP_N_BATCH=512
LLAMACPP_N_GPU_LAYERS=35
LLAMACPP_TEMPERATURE=0.2
LLAMACPP_RAG_SEARCH_K=5
LLAMACPP_MAX_CONTEXT_LENGTH=4000
```

### Максимальное качество (.env.quality)
```bash
LLAMACPP_N_CTX=8192
LLAMACPP_N_BATCH=256
LLAMACPP_N_GPU_LAYERS=30
LLAMACPP_TEMPERATURE=0.3
LLAMACPP_RAG_SEARCH_K=8
LLAMACPP_MAX_CONTEXT_LENGTH=6000
```

## 📈 Ожидаемые результаты

### Baseline (текущие настройки):
- ~15-20 tokens/sec на RTX 3070
- TTFT: ~2-3 секунды
- RAG поиск: ~300-500ms

### После оптимизации:
- **Скорость**: ~25-35 tokens/sec (+50-75%)
- **TTFT**: ~1-1.5 секунды (-50%)
- **RAG поиск**: ~150-250ms (-50%)

## 🔧 Автоматическая оптимизация

### Скрипт auto-tune
```python
# auto_optimize.py
def benchmark_settings():
    """Автоматический подбор оптимальных настроек"""
    # Тестирование разных n_batch, n_gpu_layers, n_ctx
    pass

def find_optimal_gpu_layers():
    """Поиск максимального количества слоев для GPU"""
    pass
```

## ⚠️ Предупреждения

1. **VRAM**: Не превышайте лимиты GPU памяти
2. **Stability**: Тестируйте изменения постепенно
3. **Quality**: Слишком агрессивная оптимизация может снизить качество
4. **Temperature**: Баланс между скоростью и креативностью

## 📝 Контрольный список

- [ ] Увеличить `n_gpu_layers` до максимума
- [ ] Оптимизировать `n_batch` для вашего GPU  
- [ ] Установить оптимальный `n_ctx`
- [ ] Уменьшить `rag_search_k` если возможно
- [ ] Настроить параметры генерации
- [ ] Добавить мониторинг производительности
- [ ] Протестировать разные квантизации
- [ ] Оптимизировать системные настройки 