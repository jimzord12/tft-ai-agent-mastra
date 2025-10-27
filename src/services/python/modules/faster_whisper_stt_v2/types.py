"""
Shared types for Faster-Whisper STT v2.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, Union

import numpy as np


DeviceLiteral = Literal["cpu", "cuda", "auto"]
ComputeLiteral = Literal["float32", "float16", "int8", "auto"]
TaskLiteral = Literal["transcribe", "translate"]


@dataclasses.dataclass(frozen=True)
class ModelKey:
    model_name: str
    device: Literal["cpu", "cuda"]
    compute_type: Literal["float32", "float16", "int8"]


class ResourceSnapshot(TypedDict, total=False):
    # GPU
    gpu_present: bool
    gpu_total_gb: float
    gpu_free_gb: float
    # RAM
    ram_total_gb: float
    ram_available_gb: float


class Estimate(TypedDict):
    resident_gb: float
    transient_gb: float


class TranscribeOptions(TypedDict, total=False):
    language: Optional[str]
    task: TaskLiteral
    vad_filter: bool
    beam_size: int
    return_meta: bool
    # If provided, used for capacity estimation to avoid conservative defaults
    duration_seconds: float
    # If True, try to decode WAV bytes fully in-memory instead of temp file
    decode_wav_bytes: bool


AudioInput = Union[str, bytes, np.ndarray]


class ResourceRejectedError(RuntimeError):
    def __init__(self, message: str, snapshot: Optional[ResourceSnapshot] = None):
        super().__init__(message)
        self.snapshot = snapshot or {}


class ModelLoadError(RuntimeError):
    pass
