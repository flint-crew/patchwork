"""Logging for patchwork.

Importing patchwork only creates its logger, at INFO, so an application that
uses it (flint, say) decides where its messages go, and sees each once. The
command line tools call ``setup_logging`` to print them to the terminal.
"""

from __future__ import annotations

import logging
from typing import ClassVar

logger = logging.getLogger("patchwork")
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())


class CustomFormatter(logging.Formatter):
    """A custom logger formatter"""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s.%(msecs)03d %(module)s - %(funcName)s: %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: f"{blue}%(levelname)s{reset} {format_str}",
        logging.INFO: f"{green}%(levelname)s{reset} {format_str}",
        logging.WARNING: f"{yellow}%(levelname)s{reset} {format_str}",
        logging.ERROR: f"{red}%(levelname)s{reset} {format_str}",
        logging.CRITICAL: f"{bold_red}%(levelname)s{reset} {format_str}",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


terminal = logging.StreamHandler()
"""The handler that prints patchwork's messages once ``setup_logging`` is called"""
terminal.setFormatter(CustomFormatter())


def setup_logging(level: int | str = logging.INFO) -> None:
    """Print patchwork's messages to the terminal, as the command line tools do, along with Python warnings.

    An application whose root logger already prints messages need not call it,
    as patchwork's messages reach that too. Calling it again only changes the
    level.

    Args:
        level (int | str, optional): The lowest level of message printed. Defaults to logging.INFO.
    """
    logging.captureWarnings(True)
    logger.setLevel(level)
    if terminal not in logger.handlers:
        logger.addHandler(terminal)
