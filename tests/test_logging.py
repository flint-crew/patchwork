from __future__ import annotations

import subprocess
import sys

from distributed import Client, LocalCluster

from patchwork.logging import logger, terminal
from patchwork.pipeline.cluster import LogOnWorkers


def run(code: str) -> str:
    """The standard error of Python code run in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stderr


def test_import_leaves_logging_to_the_application() -> None:
    """An application whose root logger prints messages sees each of patchwork's once, and its warnings are left alone."""
    stderr = run(
        "import logging, warnings\n"
        "logging.basicConfig(level=logging.INFO, format='ROOT %(message)s')\n"
        "import patchwork.mosaic, patchwork.pipeline.flow\n"
        "from patchwork.logging import logger\n"
        "logger.info('one message')\n"
        "warnings.warn('a warning', stacklevel=1)\n"
    )
    assert stderr.count("one message") == 1
    assert "ROOT one message" in stderr
    assert not any(
        "ROOT" in line for line in stderr.splitlines() if "a warning" in line
    )


def test_setup_logging_prints_once() -> None:
    stderr = run(
        "from patchwork.logging import logger, setup_logging\n"
        "setup_logging()\n"
        "setup_logging()\n"
        "logger.info('one message')\n"
        "logger.debug('hidden')\n"
    )
    assert stderr.count("one message") == 1
    assert "hidden" not in stderr


def test_log_on_workers() -> None:
    with (
        LocalCluster(
            n_workers=1, processes=False, threads_per_worker=1, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        client.register_plugin(LogOnWorkers())
        assert terminal in logger.handlers
    logger.removeHandler(terminal)
