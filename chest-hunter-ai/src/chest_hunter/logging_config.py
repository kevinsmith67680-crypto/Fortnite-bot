"""Logging setup for the chest_hunter package.

Log records go to stderr so that command output on stdout stays clean and
scriptable. :func:`configure_logging` is safe to call repeatedly: it replaces
the handler it installed previously instead of stacking duplicates.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

PACKAGE_LOGGER = "chest_hunter"
VALID_LEVELS: tuple[str, ...] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_HANDLER_MARKER = "_chest_hunter_handler"
LOG_FORMATS: tuple[str, ...] = ("text", "json")


class JsonFormatter(logging.Formatter):
    """One JSON object per line (Phase 7 structured logging): time (UTC, ISO 8601), level,
    logger, message, and the exception text when there is one."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": datetime.fromtimestamp(record.created, UTC).isoformat(
                     timespec="milliseconds"),
                 "level": record.levelname, "logger": record.name,
                 "message": record.getMessage(), "process": record.process,
                 "thread": record.threadName}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None,
                      log_format: str = "text") -> logging.Logger:
    """Configure and return the package logger.

    :param level: one of :data:`VALID_LEVELS` (case-insensitive).
    :param stream: destination stream; defaults to the current ``sys.stderr``.
    :raises ValueError: if ``level`` is not a recognised level name.
    """
    normalised = level.strip().upper()
    if normalised not in VALID_LEVELS:
        raise ValueError(f"Unknown log level {level!r}; expected one of {', '.join(VALID_LEVELS)}.")

    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    if log_format not in LOG_FORMATS:
        raise ValueError(f"log format must be one of {', '.join(LOG_FORMATS)}")
    handler.setFormatter(JsonFormatter() if log_format == "json"
                         else logging.Formatter(LOG_FORMAT))
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)
    logger.setLevel(normalised)
    logger.propagate = False
    return logger
