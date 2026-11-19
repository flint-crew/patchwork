"""Allows the ``patchwork`` CLI to be run with ``python -m patchwork``"""

from __future__ import annotations

from patchwork.cli import cli

if __name__ == "__main__":
    cli()
