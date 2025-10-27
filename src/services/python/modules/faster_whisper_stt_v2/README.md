# Faster Whisper STT v2

A modular, async‑ready wrapper around faster‑whisper for running speech‑to‑text with:

- Per‑config singleton model registry (no repeated heavy loads)
- Resource‑aware admission control (GPU/RAM checks + concurrency caps)
- Async facade for FastAPI (runs blocking work in a threadpool)
- Flexible audio input (path, bytes, ndarray) with optional in‑memory WAV decode

Use this when you want robust, production‑oriented STT behavior with predictable resource use.

## Contents

- Core service: `src/services/python/modules/faster_whisper_stt_v2/service.py`
- Registry (singleton cache): `src/services/python/modules/faster_whisper_stt_v2/registry.py`
- Resource manager (admission + heuristics): `src/services/python/modules/faster_whisper_stt_v2/resources.py`
- Audio I/O helpers: `src/services/python/modules/faster_whisper_stt_v2/audio_io.py`
- Transcription core: `src/services/python/modules/faster_whisper_stt_v2/transcription.py`
- Config/tuning: `src/services/python/modules/faster_whisper_stt_v2/config.py`
- Types and exceptions: `src/services/python/modules/faster_whisper_stt_v2/types.py`
- Example FastAPI app: `src/services/python/apps/stt_api.py`

## Install

Dependencies you’ll typically need (pin versions to your environment):

- faster-whisper
- fastapi, uvicorn
- torch (for CUDA detection and as a fallback GPU probe)
- psutil (RAM probing)
- pynvml (optional, more accurate VRAM probing)

Example (CPU‑only minimal):

```
pip install faster-whisper fastapi uvicorn psutil
```

GPU environments vary widely; install torch and CUDA packages matching your system.

## Quick Start (Python)

Create the async service once and reuse it:

```python
from src.services.python.modules.faster_whisper_stt_v2 import STTService

service = STTService()

# inside an async function
result = await service.transcribe_async(
    model_name="base",
    device="auto",            # auto -> cuda if available else cpu
    compute_type="auto",      # auto -> float16 on cuda, float32 on cpu
    audio_input=audio_bytes,   # bytes | path str | np.ndarray(float32, mono)
    options={
        "return_meta": True,
        "vad_filter": True,
        "beam_size": 5,
        "task": "transcribe",   # or "translate"
        "language": None,        # let model auto-detect
        "duration_seconds": 60.0,# capacity hint
        "decode_wav_bytes": True # in‑memory WAV decode
    },
)

print(result)
```

## FastAPI Example

An example API is included: `src/services/python/apps/stt_api.py`

Run it:

```
uvicorn src.services.python.apps.stt_api:app --host 0.0.0.0 --port 8000
```

Call it (multipart upload):

```
curl -X POST "http://localhost:8000/transcribe?model_name=base&device=auto&compute_type=auto&return_meta=true&decode_wav_bytes=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav"
```

## Design Overview

- Registry: Single source of truth for models, keyed by resolved `(model_name, device, compute_type)`.
  - Avoids reloading the same model across requests.
  - Per‑model semaphore caps concurrency to reduce OOM risk.
- Resource manager: Estimates memory for resident weights + transient per‑request usage and rejects when capacity is insufficient.
  - Heuristics live in `config.py` and can be tuned without changing core code.
- Async facade: `STTService` checks capacity, acquires model, and runs transcription in a threadpool.

## Tuning and Configuration

- Model size table and multipliers: `src/services/python/modules/faster_whisper_stt_v2/config.py`
  - `MODEL_RESIDENT_GB`: rough resident memory per model size
  - `COMPUTE_MULTIPLIER`: precision multipliers (float32/float16/int8)
  - `TRANSIENT_PER_MIN_GB`: transient per‑minute cost per model
  - Margins: `GPU_VRAM_MARGIN_GB`, `CPU_RAM_MARGIN_GB`
- Capacity hint:
  - Pass `duration_seconds` in `options` to improve estimation (avoid conservative defaults).
- Audio handling:
  - `decode_wav_bytes=True` tries in‑memory decode for WAV PCM; non‑WAV falls back to temp file.
- Concurrency:
  - Automatic per‑model semaphore sizing derives from free capacity and estimated transient usage.

## Error Handling

- Capacity denials raise `ResourceRejectedError` internally; the example FastAPI returns 503 with details.
- Transcription errors are wrapped as runtime errors with message context.

## Notes

- Multiple worker processes (Gunicorn/Uvicorn workers) each maintain their own model cache and consume additional memory; size your host accordingly.
- Large models (e.g., large‑v3) and higher precision consume a lot of VRAM/RAM. Prefer float16 on GPU or int8 when acceptable.
- For the original single‑class implementation, see: `src/services/python/modules/faster_whisper_stt/transcriber.py`.

## File Map

- Service facade: `src/services/python/modules/faster_whisper_stt_v2/service.py:1`
- Registry: `src/services/python/modules/faster_whisper_stt_v2/registry.py:1`
- Resource manager: `src/services/python/modules/faster_whisper_stt_v2/resources.py:1`
- Audio I/O: `src/services/python/modules/faster_whisper_stt_v2/audio_io.py:1`
- Transcription core: `src/services/python/modules/faster_whisper_stt_v2/transcription.py:1`
- Config: `src/services/python/modules/faster_whisper_stt_v2/config.py:1`
- Types: `src/services/python/modules/faster_whisper_stt_v2/types.py:1`
- Example API: `src/services/python/apps/stt_api.py:1`
