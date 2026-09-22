"""Centralised, dependency-free logging for the DataPulse Engine.

A single root logger (``datapulse``) is configured once; every module obtains a
child logger through :func:`get_logger`, so log levels can be tuned globally::

    from datapulse.logger import configure_logging
    configure_logging(level="DEBUG")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final, Optional, Union

_ROOT_LOGGER_NAME: Final[str] = "datapulse"
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"

__all__ = ["configure_logging", "get_logger"]


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colour when the stream is a TTY."""

    _COLOURS: Final[dict[int, str]] = {
        logging.DEBUG: "\033[38;5;245m",
        logging.INFO: "\033[38;5;39m",
        logging.WARNING: "\033[38;5;214m",
        logging.ERROR: "\033[38;5;196m",
        logging.CRITICAL: "\033[1;38;5;196m",
    }
    _RESET: Final[str] = "\033[0m"

    def __init__(self, *, use_colour: bool) -> None:
        super().__init__(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102 - inherited
        message = super().format(record)
        if not self._use_colour:
            return message
        colour = self._COLOURS.get(record.levelno, "")
        return f"{colour}{message}{self._RESET}"


def configure_logging(
    level: Union[int, str] = logging.INFO,
    *,
    log_file: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure and return the package root logger.

    Parameters
    ----------
    level:
        Logging level, as an ``int`` or a level name such as ``"DEBUG"``.
    log_file:
        Optional path; when provided, logs are mirrored to this file.
    force:
        Re-configure even if handlers already exist.

    Returns
    -------
    logging.Logger
        The configured ``datapulse`` root logger.

    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if logger.handlers and not force:
        logger.setLevel(level)
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(
        _ColourFormatter(use_colour=bool(getattr(sys.stdout, "isatty", lambda: False)()))
    )
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package root.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    """
    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
