"""
ResourceManager: admission control and sizing heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from . import config
from .hw_probe import gpu_memory_gb, ram_gb, resolve_auto_device_compute
from .types import Estimate, ResourceRejectedError, ResourceSnapshot


@dataclass
class ResourceManager:
    gpu_margin_gb: float = config.GPU_VRAM_MARGIN_GB
    ram_margin_gb: float = config.CPU_RAM_MARGIN_GB

    def probe(self) -> ResourceSnapshot:
        snap: ResourceSnapshot = {}
        g = gpu_memory_gb()
        if g is not None:
            snap["gpu_present"] = True
            snap["gpu_total_gb"], snap["gpu_free_gb"] = g
        else:
            snap["gpu_present"] = False

        r = ram_gb()
        if r is not None:
            snap["ram_total_gb"], snap["ram_available_gb"] = r
        return snap

    def estimate(self, model_name: str, compute_type: str, audio_minutes: float, beam_size: int) -> Estimate:
        base_resident = config.MODEL_RESIDENT_GB.get(model_name, 2.0)
        precision_mult = config.COMPUTE_MULTIPLIER.get(compute_type, 1.0)
        resident_gb = base_resident * precision_mult

        base_transient = config.TRANSIENT_PER_MIN_GB.get(model_name, 0.3)
        # Scale with minutes and beam size
        beam_scale = max(1.0, beam_size / float(config.DEFAULT_BEAM_BASELINE))
        transient_gb = max(0.1, base_transient * max(0.2, audio_minutes) * beam_scale)
        return {"resident_gb": resident_gb, "transient_gb": transient_gb}

    def can_accept(self, device: str, estimate: Estimate, is_loaded: bool, snapshot: Optional[ResourceSnapshot] = None) -> Tuple[bool, Optional[str]]:
        snap = snapshot or self.probe()
        resident = 0.0 if is_loaded else estimate["resident_gb"]
        transient = estimate["transient_gb"]

        if device == "cuda":
            if not snap.get("gpu_present"):
                return False, "GPU not present"
            free_vram = max(0.0, snap.get("gpu_free_gb", 0.0) - self.gpu_margin_gb)
            need = resident + transient
            if need <= free_vram:
                return True, None
            return False, f"Insufficient VRAM: need ~{need:.2f}GB, free ~{free_vram:.2f}GB"
        else:
            free_ram = max(0.0, snap.get("ram_available_gb", 0.0) - self.ram_margin_gb)
            need = resident + transient
            if need <= free_ram:
                return True, None
            return False, f"Insufficient RAM: need ~{need:.2f}GB, free ~{free_ram:.2f}GB"

    def concurrency_hint(self, device: str, estimate: Estimate, snapshot: Optional[ResourceSnapshot] = None) -> int:
        snap = snapshot or self.probe()
        transient = max(estimate["transient_gb"], 0.1)
        if device == "cuda":
            if not snap.get("gpu_present"):
                return 0
            free_vram = max(0.0, snap.get("gpu_free_gb", 0.0) - self.gpu_margin_gb)
            if transient <= 0.0:
                return config.DEFAULT_GPU_CONCURRENCY
            return max(1, int(free_vram // transient)) or config.DEFAULT_GPU_CONCURRENCY
        else:
            free_ram = max(0.0, snap.get("ram_available_gb", 0.0) - self.ram_margin_gb)
            if transient <= 0.0:
                return config.DEFAULT_CPU_CONCURRENCY
            return max(1, int(free_ram // transient)) or config.DEFAULT_CPU_CONCURRENCY

    def admit_or_raise(
        self,
        *,
        device: str,
        model_name: str,
        compute_type: str,
        audio_minutes: float,
        beam_size: int,
        is_loaded: bool,
    ) -> Estimate:
        est = self.estimate(model_name, compute_type, audio_minutes, beam_size)
        ok, reason = self.can_accept(device, est, is_loaded)
        if not ok:
            snap = self.probe()
            raise ResourceRejectedError(reason or "Insufficient resources", snap)
        return est

    @staticmethod
    def resolve(device: str, compute_type: str) -> Tuple[str, str]:
        return resolve_auto_device_compute(device, compute_type)

