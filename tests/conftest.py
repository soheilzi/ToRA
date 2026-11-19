import os
import random

import pytest
import torch

from tora.data.loaders import load_examples
from tora.data.prepare import data_config
from tora.data.schema import Example


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail(
            "GPU tests need CUDA_VISIBLE_DEVICES set to a free GPU on this shared server "
            "(see `python -m tora.utils.gpu_info`)"
        )
    return torch.device("cuda")


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-row relative L2 error over the last dim, in fp32."""
    x, ref = x.float(), ref.float()
    return (x - ref).norm(dim=-1) / ref.norm(dim=-1)


def aegis2_sample(n: int, seed: int) -> list[Example]:
    """n random Aegis 2.0 train examples with a response label (downloads the dataset on first use)."""
    pool = [e for e in load_examples(data_config("aegis2"), None) if e.response_label is not None and e.split == "train"]
    return random.Random(seed).sample(pool, n)
