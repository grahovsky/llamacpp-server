# 🚀 Установка GPU поддержки для llamacpp-server

## Варианты установки

### 🖥️ CPU версия (по умолчанию)
```bash
uv sync --extra cpu
```

### 🎮 GPU версии (готовые бинарники)

#### CUDA 12.1
```bash
uv sync --extra cpu
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

#### CUDA 12.2  
```bash
uv sync --extra cpu
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

#### CUDA 12.3
```bash
uv sync --extra cpu
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
```

#### CUDA 12.4 (рекомендуется)
```bash
uv sync --extra cpu
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

#### CUDA 12.5 (самая новая)
```bash
uv sync --extra cpu  
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

#### Apple Metal (macOS)
```bash
uv sync --extra cpu
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

### 🛠️ Разработка с GPU
```bash
uv sync --extra all
```

## Проверка установки

Проверьте что GPU поддержка работает:
```bash
uv run python -c "
import llama_cpp
print('llama-cpp-python version:', llama_cpp.__version__)
from llama_cpp import llama_cpp
gpu_support = hasattr(llama_cpp, 'llama_supports_gpu_offload') and llama_cpp.llama_supports_gpu_offload()
print('CUDA support:', gpu_support)
"
```

## Настройка переменных окружения

После установки GPU версии настройте переменные:

```bash
# Количество слоев на GPU (0 = CPU, -1 = все слои)
export LLAMACPP_N_GPU_LAYERS=20

# ID основной GPU (если несколько GPU)
export LLAMACPP_MAIN_GPU=0

# Разделение нагрузки между GPU (опционально)
export LLAMACPP_TENSOR_SPLIT="0.7,0.3"
```

## Диагностика проблем

### Определение версии CUDA
```bash
# Проверка установленной CUDA
nvidia-smi
nvcc --version
```

### Выбор правильной версии
- **CUDA 12.1** → cu121
- **CUDA 12.2** → cu122  
- **CUDA 12.3** → cu123
- **CUDA 12.4** → cu124 (рекомендуется)
- **CUDA 12.5** → cu125

### Проверка CUDA
```bash
nvidia-smi
```

### Проверка устройств
```bash
uv run python -c "
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    print('GPU devices:')
    print(result.stdout)
except:
    print('nvidia-smi не найден')
"
```

### Переустановка
Если что-то пошло не так:
```bash
uv remove llama-cpp-python
# Попробуйте готовые бинарники (рекомендуется)
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
# Или компиляция (нужен CUDA Toolkit)
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --force-reinstall --no-binary llama-cpp-python
```

### Установка CUDA Toolkit (для компиляции)
Если готовые бинарники не работают, нужен CUDA Toolkit:

#### Arch Linux
```bash
sudo pacman -S cuda cudnn
```

#### Ubuntu/Debian
```bash
# Добавить репозиторий NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8
```

## Запуск сервера

```bash
# С GPU (рекомендуется)
LLAMACPP_N_GPU_LAYERS=20 uv run python -m llamacpp_server.main

# Только CPU
LLAMACPP_N_GPU_LAYERS=0 uv run python -m llamacpp_server.main
```

## Производительность

- **CPU**: ~2-5 токенов/сек
- **GPU (RTX 3060)**: ~15-25 токенов/сек  
- **GPU (RTX 4090)**: ~50-80 токенов/сек

Количество слоев на GPU влияет на скорость:
- `n_gpu_layers=0` - только CPU
- `n_gpu_layers=20` - частично на GPU
- `n_gpu_layers=-1` - вся модель на GPU

## Если нужна самая свежая версия

### Напрямую через pip с нужными флагами
uv add "llama-cpp-python[cu12]" --reinstall-package

### Или для сборки из исходников (долго, но свежие функции)
CMAKE_ARGS="-DGGML_CUDA=on" uv add llama-cpp-python --force-reinstall --no-deps