from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    """One unified example. Labels: 1 = harmful, 0 = safe, None = the source has no such label."""

    id: str
    prompt: str
    response: str | None
    prompt_label: int | None
    response_label: int | None
    source: str
    split: str
    meta: dict[str, Any] = field(default_factory=dict)  # source-specific extras (e.g. adversarial flag)

    def __post_init__(self) -> None:
        for name in ("prompt_label", "response_label"):
            if getattr(self, name) not in (0, 1, None):
                raise ValueError(f"{self.id}: {name}={getattr(self, name)!r}, expected 0, 1 or None")
        if self.response is None and self.response_label is not None:
            raise ValueError(f"{self.id}: response_label set but response is None")
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: split={self.split!r}, expected one of {SPLITS}")
