"""Shared logging setup: timestamped lines on stderr, level from LOG_LEVEL (default INFO)."""

import logging
import os


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
