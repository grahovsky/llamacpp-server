"""Главный entry point приложения."""

import uvicorn

from llamacpp_server.config import get_settings
from llamacpp_server.web import create_app


def main() -> None:
    """Запуск приложения."""
    settings = get_settings()
    app = create_app()

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,  # Используем наш structlog
    )


if __name__ == "__main__":
    main()
