"""Hugging Face access: one cache root (configurable), token from $HF_TOKEN, loud errors for gated assets."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

MODEL_FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"]


def resolve_hf_home(hf_home: str | None = None) -> Path:
    if hf_home:
        return Path(hf_home).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser()
    return Path.home() / ".cache" / "huggingface"


def hub_cache_dir(hf_home: str | None = None) -> Path:
    return resolve_hf_home(hf_home) / "hub"


def datasets_cache_dir(hf_home: str | None = None) -> Path:
    return resolve_hf_home(hf_home) / "datasets"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def access_error(repo_id: str, err: Exception) -> RuntimeError:
    token_state = "HF_TOKEN is set" if hf_token() else "HF_TOKEN is NOT set"
    return RuntimeError(
        f"Cannot access {repo_id!r} ({type(err).__name__}; {token_state}). If the repo is gated, accept its "
        f"license on huggingface.co with the account that owns HF_TOKEN, then retry."
    )


def download_model(repo_id: str, revision: str | None = None, hf_home: str | None = None) -> Path:
    try:
        path = snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=hub_cache_dir(hf_home),
            token=hf_token(),
            allow_patterns=MODEL_FILE_PATTERNS,
        )
    except (GatedRepoError, RepositoryNotFoundError) as err:
        raise access_error(repo_id, err) from err
    return Path(path)
