"""Filesystem anchors. The package is installed editable, so these point into the source tree."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_ROOT = CODE_ROOT / "configs"
RESULTS_ROOT = CODE_ROOT / "results"


def resolve(path: str | Path, base: Path = CODE_ROOT) -> Path:
    """Absolute paths pass through; relative paths are taken relative to `base`."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else base / path
