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
        
        assert settings.n_ctx == 2048
        assert settings.n_batch == 512
        assert settings.host == "0.0.0.0"
        assert settings.port == 8090
        assert settings.temperature == 0.7
        assert settings.dev_mode is True


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
        'LLAMACPP_N_CTX': '4096'
    }):
        settings = Settings()
        
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.n_ctx == 4096


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