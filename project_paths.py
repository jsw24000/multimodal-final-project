"""One configurable root for PEARL datasets and large artifacts."""

import os
import shutil
from pathlib import Path


DATA_ROOT = Path(os.environ.get("PEARL_DATA_ROOT", "/data/Multimodal-final-project")).expanduser()


def require_large_disk() -> None:
    """Fail before model or output creation if the data root is unsafe."""
    if not DATA_ROOT.is_dir():
        raise RuntimeError(f"PEARL_DATA_ROOT does not exist: {DATA_ROOT}")
    if DATA_ROOT.stat().st_dev == Path("/").stat().st_dev:
        raise RuntimeError(f"PEARL_DATA_ROOT is on the system filesystem: {DATA_ROOT}")
    if shutil.disk_usage(DATA_ROOT).total < 6_000_000_000_000:
        raise RuntimeError(f"PEARL_DATA_ROOT is not on the expected large disk: {DATA_ROOT}")
    if not os.access(DATA_ROOT, os.W_OK):
        raise RuntimeError(f"PEARL_DATA_ROOT is not writable: {DATA_ROOT}")


def dataset_path(*parts: str) -> str:
    return str(DATA_ROOT.joinpath("datasets", *parts))


def output_path(*parts: str) -> str:
    require_large_disk()
    return str(DATA_ROOT.joinpath("outputs", *parts))


def checkpoint_path(*parts: str) -> str:
    require_large_disk()
    return str(DATA_ROOT.joinpath("checkpoints", *parts))
