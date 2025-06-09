"""Infrastructure layer - логирование, DI контейнер."""

from .container import Container
from .logging import setup_logging

__all__ = ["Container", "setup_logging"]
