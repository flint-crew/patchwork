from __future__ import annotations

import importlib.metadata

import patchwork as m


def test_version() -> None:
    assert importlib.metadata.version("askap-patchwork") == m.__version__
