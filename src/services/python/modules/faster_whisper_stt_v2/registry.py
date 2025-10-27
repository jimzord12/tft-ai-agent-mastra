"""
WhisperModelRegistry: singleton cache with async-safe creation and per-model
concurrency control.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

from faster_whisper import WhisperModel

from .hw_probe import resolve_auto_device_compute
from .types import ModelKey


class WhisperModelRegistry:
    def __init__(self):
        self._models: Dict[ModelKey, WhisperModel] = {}
        self._locks: Dict[ModelKey, asyncio.Lock] = {}
        self._semaphores: Dict[ModelKey, asyncio.Semaphore] = {}

    @staticmethod
    def _key(model_name: str, device: str, compute_type: str) -> ModelKey:
        d, c = resolve_auto_device_compute(device, compute_type)
        if d not in ("cpu", "cuda"):
            d = "cpu"
        if c not in ("float32", "float16", "int8"):
            c = "float32" if d == "cpu" else "float16"
        return ModelKey(model_name=model_name, device=d, compute_type=c)

    def is_loaded(self, model_name: str, device: str, compute_type: str) -> bool:
        key = self._key(model_name, device, compute_type)
        return key in self._models

    async def get_or_create(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        *,
        concurrency: Optional[int] = None,
    ) -> Tuple[ModelKey, WhisperModel]:
        key = self._key(model_name, device, compute_type)
        if key in self._models:
            return key, self._models[key]

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._models:
                return key, self._models[key]
            model = WhisperModel(key.model_name, device=key.device, compute_type=key.compute_type)
            self._models[key] = model
            # default semaphore; may be overridden by service based on resources
            self._semaphores.setdefault(key, asyncio.Semaphore(concurrency or 1))
            return key, model

    def get_semaphore(self, key: ModelKey) -> asyncio.Semaphore:
        return self._semaphores.setdefault(key, asyncio.Semaphore(1))

