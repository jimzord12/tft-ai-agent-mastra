"""
Async STT service facade that combines ResourceManager and WhisperModelRegistry.
Suitable for FastAPI endpoints.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Union

from fastapi.concurrency import run_in_threadpool

from .resources import ResourceManager
from .registry import WhisperModelRegistry
from .transcription import transcribe_with_model
from .types import AudioInput, ModelKey, ResourceRejectedError, TranscribeOptions


class STTService:
    def __init__(self, registry: Optional[WhisperModelRegistry] = None, resources: Optional[ResourceManager] = None):
        self.registry = registry or WhisperModelRegistry()
        self.resources = resources or ResourceManager()

    @staticmethod
    def _estimate_audio_minutes(audio_input: AudioInput) -> float:
        # Without decoding, use a conservative small default. You can replace this
        # by inspecting headers or passing duration in options.
        return 1.0

    async def transcribe_async(
        self,
        *,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        audio_input: AudioInput,
        options: Optional[TranscribeOptions] = None,
    ) -> Union[str, Dict[str, Any]]:
        opts: TranscribeOptions = {
            "language": None,
            "task": "transcribe",
            "vad_filter": True,
            "beam_size": 5,
            "return_meta": False,
            "decode_wav_bytes": True,
        }
        if options:
            opts.update(options)

        # Check resources and compute concurrency hint
        resolved_device, resolved_compute = self.resources.resolve(device, compute_type)
        is_loaded = self.registry.is_loaded(model_name, resolved_device, resolved_compute)
        if isinstance(opts.get("duration_seconds"), (int, float)) and float(opts["duration_seconds"]) > 0:
            audio_minutes = float(opts["duration_seconds"]) / 60.0
        else:
            audio_minutes = self._estimate_audio_minutes(audio_input)
        est = self.resources.admit_or_raise(
            device=resolved_device,
            model_name=model_name,
            compute_type=resolved_compute,
            audio_minutes=audio_minutes,
            beam_size=int(opts.get("beam_size", 5) or 5),
            is_loaded=is_loaded,
        )

        # Get or create model and set per-model semaphore capacity
        key, model = await self.registry.get_or_create(
            model_name=model_name,
            device=resolved_device,
            compute_type=resolved_compute,
            concurrency=max(1, self.resources.concurrency_hint(resolved_device, est)),
        )

        sem = self.registry.get_semaphore(key)
        async with sem:
            return await run_in_threadpool(
                transcribe_with_model,
                model,
                audio_input,
                language=opts.get("language"),
                task=opts.get("task", "transcribe"),
                vad_filter=bool(opts.get("vad_filter", True)),
                beam_size=int(opts.get("beam_size", 5) or 5),
                return_meta=bool(opts.get("return_meta", False)),
                decode_wav_bytes=bool(opts.get("decode_wav_bytes", True)),
            )
