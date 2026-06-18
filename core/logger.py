"""
core/logger.py
Configures system-wide logging to both console and a rotating log file.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(log_dir: str = "logs/", log_level: str = "INFO") -> logging.Logger:
    """
    Set up the root logger with:
    - Console handler (INFO+)
    - Rotating file handler (DEBUG+), max 5MB per file, 3 backups
    Returns the root logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir, f"auth_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-init
    if root_logger.handlers:
        root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # --- File handler ---
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Logger initialized. Log file: {log_filename}")
    return root_logger
