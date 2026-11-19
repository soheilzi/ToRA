"""NVML snapshot of every GPU on the box, which of them this process may use, and who else is on them.

Run `python -m tora.utils.gpu_info` to print the snapshot and the GPUs that currently look free.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pynvml


@dataclass
class GPUProcess:
    pid: int
    used_mb: float | None
    name: str | None


@dataclass
class GPUState:
    index: int  # physical index (nvidia-smi numbering)
    uuid: str
    name: str
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: int
    visible: bool  # listed in CUDA_VISIBLE_DEVICES (or it is unset)
    processes: list[GPUProcess] = field(default_factory=list)


def visible_device_spec() -> list[str] | None:
    spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if spec is None:
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


def require_visible_devices() -> list[str]:
    """Entry points call this so nothing runs on a shared box without an explicit GPU choice."""
    spec = visible_device_spec()
    if not spec:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is not set. This is a shared server: pick free GPU(s) "
            "(see `python -m tora.utils.gpu_info`) and run e.g. `CUDA_VISIBLE_DEVICES=0 ...`."
        )
    return spec


def _is_visible(index: int, uuid: str, spec: list[str] | None) -> bool:
    if spec is None:
        return True
    return any(s == str(index) or (s.startswith(("GPU-", "MIG-")) and uuid.startswith(s)) for s in spec)


def _str(x: bytes | str) -> str:
    return x.decode() if isinstance(x, bytes) else x


def _process_name(pid: int) -> str | None:
    try:
        return _str(pynvml.nvmlSystemGetProcessName(pid))
    except pynvml.NVMLError:
        return None


def query_gpus() -> list[GPUState]:
    pynvml.nvmlInit()
    try:
        spec = visible_device_spec()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = _str(pynvml.nvmlDeviceGetUUID(h))
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            procs = [
                GPUProcess(
                    pid=p.pid,
                    used_mb=None if p.usedGpuMemory is None else p.usedGpuMemory / 2**20,
                    name=_process_name(p.pid),
                )
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            ]
            gpus.append(
                GPUState(
                    index=i,
                    uuid=uuid,
                    name=_str(pynvml.nvmlDeviceGetName(h)),
                    memory_used_mb=mem.used / 2**20,
                    memory_total_mb=mem.total / 2**20,
                    utilization_pct=pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                    visible=_is_visible(i, uuid, spec),
                    processes=procs,
                )
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def snapshot() -> dict:
    """JSON-serialisable snapshot; logged with every run and every benchmark."""
    gpus = query_gpus()
    own = os.getpid()
    others = [
        {"gpu": g.index, "pid": p.pid, "used_mb": p.used_mb, "name": p.name}
        for g in gpus
        if g.visible
        for p in g.processes
        if p.pid != own
    ]
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [asdict(g) for g in gpus],
        "other_processes_on_visible_gpus": others,
        "visible_gpus_shared": bool(others),
    }


def free_gpus(max_used_mb: float = 1024.0) -> list[int]:
    """Physical indices with no compute processes and little memory in use."""
    return [g.index for g in query_gpus() if not g.processes and g.memory_used_mb <= max_used_mb]


def format_snapshot(snap: dict) -> str:
    lines = [f"{snap['hostname']}  CUDA_VISIBLE_DEVICES={snap['cuda_visible_devices']}"]
    for g in snap["gpus"]:
        procs = ", ".join(f"{p['pid']}:{p['name'] or '?'}" for p in g["processes"]) or "-"
        mark = "*" if g["visible"] else " "
        lines.append(
            f"{mark} GPU{g['index']} {g['name']}  {g['memory_used_mb']:.0f}/{g['memory_total_mb']:.0f} MiB  "
            f"util {g['utilization_pct']}%  procs: {procs}"
        )
    if snap["visible_gpus_shared"]:
        lines.append(f"WARNING: other processes on visible GPUs: {snap['other_processes_on_visible_gpus']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_snapshot(snapshot()))
    print("free:", ",".join(map(str, free_gpus())) or "none")
