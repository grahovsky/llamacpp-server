"""Модуль конфигурации приложения."""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]


def diagnose_devices() -> dict:
    """Диагностика доступных устройств для вычислений."""
    settings = get_settings()
    return settings.device_info
