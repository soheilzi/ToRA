"""Run records: every run writes results/<phase>/<run_id>.jsonl.

Line 1 is a `meta` record (config, git, seed, GPU snapshot, library versions); then `metrics`
records; then an `end` record. wandb mirroring is optional and fails loudly if requested but missing.
"""

from __future__ import annotations

import json
import logging
import platform
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tora.utils import gpu_info
from tora.utils.paths import CODE_ROOT, resolve


def console_logger(name: str = "tora") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def git_info(repo_dir: Path = CODE_ROOT) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        status = git("status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"hash": None, "dirty": None, "note": "not a git repository"}
    try:
        commit = git("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return {"hash": None, "dirty": True, "note": "no commits yet"}
    return {"hash": commit, "dirty": bool(status)}


def library_versions() -> dict:
    import torch
    import transformers
    import triton

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "transformers": transformers.__version__,
    }


def make_run_id(name: str) -> str:
    return f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy / torch scalars
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


class RunLogger:
    def __init__(
        self,
        phase: str,
        run_id: str,
        config: dict,
        seed: int,
        results_dir: str | Path = "results",
        wandb_project: str | None = None,
    ):
        self.phase, self.run_id = phase, run_id
        self.path = resolve(results_dir) / phase / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        self._t0 = time.perf_counter()
        self.console = console_logger()
        self._wandb = None
        if wandb_project:
            try:
                import wandb
            except ImportError as err:
                raise ImportError("run.wandb=true but wandb is not installed (pip install wandb)") from err
            self._wandb = wandb.init(project=wandb_project, name=run_id, config=config)

        gpu = gpu_info.snapshot()
        self._write(
            {
                "type": "meta",
                "run_id": run_id,
                "phase": phase,
                "time": datetime.now().isoformat(),
                "argv": sys.argv,
                "config": config,
                "seed": seed,
                "git": git_info(),
                "versions": library_versions(),
                "gpu": gpu,
            }
        )
        self.console.info("run %s -> %s", run_id, self.path)
        self.console.info("GPU snapshot:\n%s", gpu_info.format_snapshot(gpu))
        if gpu["visible_gpus_shared"]:
            self.console.warning("other processes share the visible GPU(s); timing results from this run are suspect")

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_jsonable) + "\n")
        self._fh.flush()

    def log(self, metrics: dict, step: int | None = None, **tags: Any) -> None:
        record = {"type": "metrics", "elapsed_s": time.perf_counter() - self._t0, **tags, **metrics}
        if step is not None:
            record["step"] = step
        self._write(record)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def gpu_snapshot(self, tag: str) -> dict:
        snap = gpu_info.snapshot()
        self._write({"type": "gpu", "tag": tag, "gpu": snap})
        return snap

    def close(self, status: str = "ok") -> None:
        if self._fh.closed:
            return
        self._write({"type": "end", "status": status, "elapsed_s": time.perf_counter() - self._t0})
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, *_) -> None:
        self.close("ok" if exc_type is None else f"error: {exc_type.__name__}")
