"""
Configuration and tunables for Faster-Whisper STT v2.

These values are heuristics and safe defaults. You can tune them
based on production telemetry without touching core logic.
"""

from __future__ import annotations

from typing import Dict

# Approximate resident memory footprint (GB) by model size.
# Values are rough and intended for admission heuristics.
MODEL_RESIDENT_GB: Dict[str, float] = {
    "tiny": 0.7,
    "base": 1.4,
    "small": 2.5,
    "medium": 5.0,
    "large": 10.0,
    "large-v2": 11.0,
    "large-v3": 12.0,
}

# Compute type multipliers (approx) applied to resident size.
COMPUTE_MULTIPLIER: Dict[str, float] = {
    "float32": 2.0,
    "float16": 1.0,
    "int8": 0.6,
}

# Transient per-inference base footprint per minute of audio (GB) by model.
TRANSIENT_PER_MIN_GB: Dict[str, float] = {
    "tiny": 0.2,
    "base": 0.2,
    "small": 0.3,
    "medium": 0.5,
    "large": 0.8,
    "large-v2": 0.9,
    "large-v3": 1.0,
}

# Default beam size baseline for transient estimates.
DEFAULT_BEAM_BASELINE = 5

# Capacity safety margins (GB) kept free in admission checks.
GPU_VRAM_MARGIN_GB = 1.5
CPU_RAM_MARGIN_GB = 2.0

# Default per-model concurrency caps if none are computed dynamically.
DEFAULT_GPU_CONCURRENCY = 1
DEFAULT_CPU_CONCURRENCY = 2

