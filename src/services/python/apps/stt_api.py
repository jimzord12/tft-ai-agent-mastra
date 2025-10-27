"""
Minimal FastAPI app wiring the async STTService.

Run locally (example):
  uvicorn src.services.python.apps.stt_api:app --host 0.0.0.0 --port 8000

POST /transcribe with form-data file upload (field name: file) and optional query params:
  - model_name: str = base
  - device: str = auto|cpu|cuda
  - compute_type: str = auto|float16|float32|int8
  - return_meta: bool = true|false
  - beam_size: int
  - vad_filter: bool
  - task: transcribe|translate
  - language: optional language code
  - duration_seconds: float (estimation hint)
  - decode_wav_bytes: bool (prefer in-memory decoding for WAV bytes)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.services.python.modules.faster_whisper_stt_v2 import STTService


app = FastAPI(title="Faster-Whisper STT API (v2)")


@app.on_event("startup")
async def _startup() -> None:
    # Create a singleton service for the process
    app.state.stt_service = STTService()


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    return_meta: bool = True,
    beam_size: int = 5,
    vad_filter: bool = True,
    task: str = "transcribe",
    language: str | None = None,
    duration_seconds: float | None = None,
    decode_wav_bytes: bool = True,
):
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    service: STTService = app.state.stt_service

    try:
        result = await service.transcribe_async(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            audio_input=audio_bytes,
            options={
                "return_meta": return_meta,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "task": task,
                "language": language,
                "duration_seconds": duration_seconds or 0.0,
                "decode_wav_bytes": decode_wav_bytes,
            },
        )
    except Exception as e:
        # resource rejections and other runtime errors bubble up here
        raise HTTPException(status_code=503, detail=str(e))

    if isinstance(result, dict):
        return JSONResponse(content=result)
    return {"text": str(result)}

