"""
Hardware probing utilities and auto-resolution helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple


def is_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False


def gpu_memory_gb() -> Optional[Tuple[float, float]]:
    """
    Returns (total_gb, free_gb) for the first visible GPU if available.
    Falls back to torch if pynvml is not installed.
    """
    # Try pynvml for accurate readings
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = mem.total / (1024 ** 3)
        free_gb = mem.free / (1024 ** 3)
        pynvml.nvmlShutdown()
        return total_gb, free_gb
    except Exception:
        pass

    # Fallback to torch if available
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory
            # Rough free estimate: total - reserved
            reserved = torch.cuda.memory_reserved(idx)
            total_gb = total / (1024 ** 3)
            free_gb = max(0.0, (total - reserved) / (1024 ** 3))
            return total_gb, free_gb
    except Exception:
        pass

    return None


def ram_gb() -> Optional[Tuple[float, float]]:
    """
    Returns (total_gb, available_gb) of system RAM using psutil if available.
    """
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
    except Exception:
        return None


def resolve_auto_device_compute(device: str, compute_type: str) -> Tuple[str, str]:
    d = device
    if d == "auto":
        d = "cuda" if is_cuda_available() else "cpu"
    c = compute_type
    if c == "auto":
        c = "float16" if d == "cuda" else "float32"
    return d, c

