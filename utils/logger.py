import sys
from loguru import logger
from pathlib import Path


def setup_logging():
    """Configure logging for the application."""

    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "lawbot.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 week"
    )

    return logger


# Initialize logger
logger = setup_logging()