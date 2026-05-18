import logging
from pathlib import Path

from rich.logging import RichHandler

from robolab.configs import cfg

current_dir = Path(__file__).resolve().parent
log_dir = current_dir.parent.parent / "logs"
log_file = log_dir / "robolab.log"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=cfg.hyperparams.logging_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("robolab")
