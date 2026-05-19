"""Logging configuration for the Robolab package.

Sets up structured logging with both Rich console output and
rotating file handlers, using configuration from the global Config singleton.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from robolab.configs import cfg

# Resolve log directory relative to project root (two levels up from this package)
current_dir = Path(__file__).resolve().parent
log_dir = current_dir.parent.parent / "logs"
log_file = log_dir / "robolab.log"
log_dir.mkdir(parents=True, exist_ok=True)

# Configure root logger with Rich console handler and rotating file handler
logging.basicConfig(
    level=cfg.hyperparams.logging_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("robolab")
