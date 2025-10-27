"""
faster_whisper_stt_v2
Modular v2 package for Faster-Whisper STT with model registry and
resource-aware admission control suitable for async servers.

Public entrypoints most users want:
- service: STTService (async facade)
- registry: WhisperModelRegistry (singleton cache)
"""

from .service import STTService  # noqa: F401
from .registry import WhisperModelRegistry  # noqa: F401

