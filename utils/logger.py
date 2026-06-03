"""Centralized loguru logger."""

from __future__ import annotations
import sys
from loguru import logger
from config.settings import settings

logger.remove()

logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    enqueue=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    settings.log_dir / "engine_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="14 days",
    compression="zip",
    enqueue=True,
    serialize=False,
)

__all__ = ["logger"]