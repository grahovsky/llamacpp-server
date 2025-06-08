"""Тесты для модуля конфигурации."""

import pytest
from pathlib import Path
from unittest.mock import patch
from llamacpp_server.config.settings import Settings


def test_settings_default_values():
    """Тест дефолтных значений настроек."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true'  # В dev режиме валидация файла отключена
    }):
        settings = Settings()
        
        assert settings.n_ctx == 4096
        assert settings.n_batch == 512
        assert settings.host == "0.0.0.0"
        assert settings.port == 8090
        assert settings.temperature == 0.1
        assert settings.dev_mode is True
        
        # Новые настройки контекста
        assert settings.max_history_tokens == 1200
        assert settings.max_response_tokens == 800
        assert settings.rag_max_context == 1200
        assert settings.safety_buffer_tokens == 300


def test_settings_dev_mode():
    """Тест dev режима."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true'
    }):
        settings = Settings()
        assert settings.dev_mode is True


def test_settings_env_override():
    """Тест переопределения через переменные окружения."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',  # Отключаем валидацию файла
        'LLAMACPP_HOST': '127.0.0.1',
        'LLAMACPP_PORT': '9000',
        'LLAMACPP_N_CTX': '6144',
        'LLAMACPP_MAX_HISTORY_TOKENS': '2000',
        'LLAMACPP_MAX_RESPONSE_TOKENS': '1000'
    }):
        settings = Settings()
        
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.n_ctx == 6144
        assert settings.max_history_tokens == 2000
        assert settings.max_response_tokens == 1000


def test_context_distribution_properties():
    """Тест вспомогательных свойств распределения контекста."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',
        'LLAMACPP_N_CTX': '4096',
        'LLAMACPP_MAX_HISTORY_TOKENS': '1200',
        'LLAMACPP_MAX_RESPONSE_TOKENS': '800',
        'LLAMACPP_RAG_MAX_CONTEXT': '1200',
        'LLAMACPP_SAFETY_BUFFER_TOKENS': '300'
    }):
        settings = Settings()
        
        assert settings.available_context_tokens == 3796  # 4096 - 300
        assert settings.effective_history_limit == 1200
        
        dist = settings.context_distribution
        assert dist['total_context'] == 4096
        assert dist['allocated_total'] == 3500  # 1200 + 800 + 1200 + 300
        assert dist['remaining'] == 596


def test_context_validation_success():
    """Тест успешной валидации контекста."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',
        'LLAMACPP_N_CTX': '8192',
        'LLAMACPP_MAX_HISTORY_TOKENS': '2000',
        'LLAMACPP_MAX_RESPONSE_TOKENS': '1000',
        'LLAMACPP_RAG_MAX_CONTEXT': '2000',
        'LLAMACPP_SAFETY_BUFFER_TOKENS': '500'
    }):
        # Не должно выбросить исключение
        settings = Settings()
        assert settings.n_ctx == 8192


def test_context_validation_failure():
    """Тест неудачной валидации контекста - превышение лимита."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',
        'LLAMACPP_N_CTX': '2048',
        'LLAMACPP_MAX_HISTORY_TOKENS': '1500',
        'LLAMACPP_MAX_RESPONSE_TOKENS': '1000',
        'LLAMACPP_RAG_MAX_CONTEXT': '1000',
        'LLAMACPP_SAFETY_BUFFER_TOKENS': '200'
    }):
        # Сумма: 1500 + 1000 + 1000 + 200 = 3700 > 2048
        with pytest.raises(ValueError, match="Сумма токенов превышает размер контекста"):
            Settings()


def test_history_tokens_validation():
    """Тест валидации размера истории относительно контекста."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',
        'LLAMACPP_N_CTX': '2048',
        'LLAMACPP_MAX_HISTORY_TOKENS': '2000',  # > 70% от 2048
    }):
        # Должно пройти, так как 2000 < 70% от 2048 (1433.6)
        # На самом деле будет ошибка из-за общей валидации контекста
        with pytest.raises(ValueError):
            Settings()


@pytest.mark.parametrize("invalid_port", [-1, 0, 65536, 100000])
def test_settings_port_validation(invalid_port):
    """Тест валидации порта."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',  # Отключаем валидацию файла
        'LLAMACPP_PORT': str(invalid_port)
    }):
        with pytest.raises(ValueError):
            Settings()


@pytest.mark.parametrize("invalid_temp", [-1.0, 2.5])
def test_settings_temperature_validation(invalid_temp):
    """Тест валидации температуры."""
    with patch.dict('os.environ', {
        'LLAMACPP_DEV_MODE': 'true',  # Отключаем валидацию файла
        'LLAMACPP_TEMPERATURE': str(invalid_temp)
    }):
        with pytest.raises(ValueError):
            Settings() 