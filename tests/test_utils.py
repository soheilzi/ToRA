import json

import pytest
import torch

from tora.data.schema import Example
from tora.utils import gpu_info
from tora.utils.runlog import RunLogger, git_info
from tora.utils.seeding import seed_everything


def test_seed_everything_reproducible():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    assert torch.equal(a, torch.randn(4))


def test_visible_device_parsing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 3")
    assert gpu_info.visible_device_spec() == ["1", "3"]
    assert gpu_info._is_visible(3, "GPU-abc", ["1", "3"])
    assert not gpu_info._is_visible(0, "GPU-abc", ["1", "3"])
    assert gpu_info._is_visible(0, "GPU-abcdef", ["GPU-abc"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert gpu_info.visible_device_spec() is None
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        gpu_info.require_visible_devices()


def test_gpu_snapshot_structure():
    snap = gpu_info.snapshot()
    assert {"time", "hostname", "cuda_visible_devices", "gpus", "other_processes_on_visible_gpus"} <= snap.keys()
    for g in snap["gpus"]:
        assert {"index", "uuid", "memory_used_mb", "processes", "visible"} <= g.keys()
    json.dumps(snap)


def test_run_logger_records(tmp_path):
    with RunLogger("unit", "run-x", {"a": 1}, seed=7, results_dir=tmp_path) as log:
        log.log({"loss": torch.tensor(0.5)}, step=3)
    records = [json.loads(line) for line in (tmp_path / "unit" / "run-x.jsonl").read_text().splitlines()]
    assert [r["type"] for r in records] == ["meta", "metrics", "end"]
    meta = records[0]
    assert meta["config"] == {"a": 1} and meta["seed"] == 7
    assert {"git", "gpu", "versions"} <= meta.keys()
    assert records[1]["loss"] == 0.5 and records[1]["step"] == 3
    assert records[2]["status"] == "ok"


def test_git_info_shape():
    info = git_info()
    assert {"hash", "dirty"} <= info.keys()


def test_example_validation():
    Example("a", "p", None, 1, None, "src", "train")
    with pytest.raises(ValueError):
        Example("a", "p", None, 1, 0, "src", "train")  # label without a response
    with pytest.raises(ValueError):
        Example("a", "p", "r", 2, 0, "src", "train")
    with pytest.raises(ValueError):
        Example("a", "p", "r", 1, 0, "src", "dev")
