"""GPU timing (check S10 hygiene): warm-up, a synchronize before every repeat, CUDA events around
it, medians reported.

A repeat is one call of a zero-argument function that reads and writes static tensors. Eager calls include their
CPU launch time (the GPU waits for it); `cuda_graph` turns such a function into a graph replay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def cuda_graph(fn: Callable[[], Any], warmup: int = 3) -> Callable[[], Any]:
    """Capture fn in a CUDA graph (after warm-up calls on a side stream, as torch.cuda.graphs recommends); the returned
    callable replays it and returns the captured outputs, which every replay overwrites in place."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()

    def replay() -> Any:
        graph.replay()
        return out

    replay.graph = graph  # keep it alive with the callable
    return replay


def time_ms(fn: Callable[[], Any], warmup: int = 20, reps: int = 100) -> dict[str, float]:
    """Latency of fn in ms over `reps` synchronized repeats: median, p10, p90, mean."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()

    def q(p: float) -> float:
        return times[min(len(times) - 1, int(p * len(times)))]

    return {"median_ms": q(0.5), "p10_ms": q(0.1), "p90_ms": q(0.9), "mean_ms": sum(times) / len(times), "reps": reps}
