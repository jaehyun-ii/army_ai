"""
Centralized logging configuration.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[91m',  # Bright Red
    }
    RESET = '\033[0m'

    def format(self, record):
        if not sys.stderr.isatty():
            return super().format(record)

        levelname = record.levelname
        if levelname in self.COLORS:
            levelname_color = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            record.levelname = levelname_color

        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    log_level = log_level or settings.LOG_LEVEL
    log_file = log_file or settings.LOG_FILE

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATE_FORMAT
        )
    else:
        console_formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATE_FORMAT
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))

        file_formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    configure_third_party_loggers()


def configure_third_party_loggers():
    """
    Configure logging levels for third-party libraries.

    Uses LOG_THIRD_PARTY_LEVELS from settings for flexible configuration.
    This allows dynamic control of third-party library verbosity without code changes.

    Environment variable example:
        LOG_THIRD_PARTY_LEVELS='{"uvicorn.access": "ERROR", "sqlalchemy": "INFO"}'
    """
    # Get third-party log levels from settings
    third_party_levels = settings.LOG_THIRD_PARTY_LEVELS or {}

    # Apply configured log levels
    for logger_name, level_str in third_party_levels.items():
        try:
            level = getattr(logging, level_str.upper())
            logging.getLogger(logger_name).setLevel(level)
        except AttributeError:
            # Invalid log level, skip
            logging.warning(f"Invalid log level '{level_str}' for logger '{logger_name}'")

    # Additional production-specific silencing (if not already configured)
    if settings.is_production():
        # Only apply if not already configured via settings
        if "PIL" not in third_party_levels:
            logging.getLogger("PIL").setLevel(logging.ERROR)
        if "urllib3" not in third_party_levels:
            logging.getLogger("urllib3").setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration changes."""

    def __init__(self, level: str):
        self.level = level
        self.original_level = None
        self.logger = logging.getLogger()

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(getattr(logging, self.level.upper()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


# Request ID context for tracing
import contextvars
request_id_context = contextvars.ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """Add request ID to log records."""

    def filter(self, record):
        record.request_id = request_id_context.get() or "N/A"
        return True


def setup_request_logging():
    """Setup request ID logging."""
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdFilter())

    # Update format to include request ID
    if "[%(request_id)s]" not in settings.LOG_FORMAT:
        new_format = settings.LOG_FORMAT.replace(
            "%(levelname)s",
            "%(levelname)s [%(request_id)s]"
        )
        for handler in logging.getLogger().handlers:
            handler.setFormatter(logging.Formatter(
                fmt=new_format,
                datefmt=settings.LOG_DATE_FORMAT
            ))