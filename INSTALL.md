# Установка uv (если еще нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Создание нового проекта с изолированным окружением
cd /home/grahovsky/projects/llamacpp-server
uv venv --python 3.11 .venv

# Активация окружения
source .venv/bin/activate

# Установка зависимостей из pyproject.toml
uv pip install -e .

# Запуск сервера
python server.py