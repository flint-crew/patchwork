"""Dask clusters that the pipeline runs on"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from distributed import LocalCluster
from distributed.diagnostics.plugin import WorkerPlugin

from patchwork.logging import logger, setup_logging

if TYPE_CHECKING:
    from distributed import Worker
    from distributed.deploy.cluster import Cluster


# Without distributed's type hints, as in pre-commit's mypy, WorkerPlugin is Any
class LogOnWorkers(WorkerPlugin):  # type: ignore[misc,unused-ignore]
    """Prints patchwork's messages on every worker of a cluster, including those that join later"""

    def __init__(self, level: int | str = logging.INFO) -> None:
        self.level = level

    def setup(self, worker: Worker) -> None:  # noqa: ARG002
        """Print patchwork's messages on the worker."""
        setup_logging(self.level)


def get_cluster_spec(cluster_config: str | Path) -> dict[str, Any]:
    """Read a cluster specification, with keys ``cluster_class``, ``cluster_kwargs`` and ``adapt_kwargs``."""
    with Path(cluster_config).open() as config_file:
        spec: dict[str, Any] = yaml.safe_load(config_file)
    if "cluster_class" not in spec:
        msg = f"{cluster_config} does not name a cluster_class"
        raise ValueError(msg)
    return spec


def get_cluster(
    cluster_config: str | Path | None = None, n_workers: int | None = None
) -> Cluster:
    """Start a dask cluster, either on this machine or from a cluster specification.

    A specification names the class of the cluster, such as
    ``dask_jobqueue.SLURMCluster``, the arguments it is created with, and
    optionally how it adapts its number of workers. Without one a local cluster
    of single threaded workers is started.

    Args:
        cluster_config (str | Path | None, optional): YAML cluster specification. Defaults to None, a local cluster.
        n_workers (int | None, optional): Number of workers, when the cluster does not adapt. Defaults to None.

    Returns:
        Cluster: The running cluster
    """
    if cluster_config is None:
        return LocalCluster(n_workers=n_workers, threads_per_worker=1)

    spec = get_cluster_spec(cluster_config)
    module_name, class_name = spec["cluster_class"].rsplit(".", 1)
    cluster_class = getattr(importlib.import_module(module_name), class_name)
    cluster: Cluster = cluster_class(**spec.get("cluster_kwargs", {}))
    logger.info(f"Started a {class_name} from {cluster_config}")

    if spec.get("adapt_kwargs"):
        cluster.adapt(**spec["adapt_kwargs"])
    elif n_workers is not None:
        cluster.scale(n_workers)
    return cluster
