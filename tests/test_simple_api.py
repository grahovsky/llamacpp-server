"""Упрощенные тесты модулей."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestModuleImports:
    """Тесты импорта модулей."""

    def test_import_main_module(self):
        """Тест импорта основного модуля."""
        from llamacpp_server import main
        assert hasattr(main, 'create_app')

    def test_import_settings(self):
        """Тест импорта настроек."""
        with patch.dict('os.environ', {'LLAMACPP_DEV_MODE': 'true'}):
            from llamacpp_server.config.settings import Settings
            settings = Settings()
            assert settings.dev_mode is True

    def test_import_domain_models(self):
        """Тест импорта domain моделей."""
        from llamacpp_server.domain.models import (
            ChatMessage,
        )

        # Проверяем создание моделей
        message = ChatMessage(role="user", content="test")
        assert message.role == "user"
        assert message.content == "test"

    def test_import_llama_service(self):
        """Тест импорта LlamaService."""
        from unittest.mock import Mock

        from llamacpp_server.llama.service import LlamaService

        mock_llama = Mock()
        service = LlamaService(mock_llama)
        assert service is not None

    def test_import_routers(self):
        """Тест импорта роутеров."""
        from llamacpp_server.web.routers import health, ollama, openai, webui

        assert hasattr(health, 'router')
        assert hasattr(webui, 'router')
        assert hasattr(ollama, 'router')
        assert hasattr(openai, 'router')
