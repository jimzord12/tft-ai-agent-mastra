"""
Pure transcription logic using a provided WhisperModel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from .audio_io import AudioInput, prepare_audio_input


def transcribe_with_model(
    model: Any,
    audio_input: AudioInput,
    *,
    language: str | None = None,
    task: str = "transcribe",
    vad_filter: bool = True,
    beam_size: int = 5,
    return_meta: bool = False,
    decode_wav_bytes: bool = True,
) -> Union[str, Dict[str, Any]]:
    """
    Runs faster-whisper transcription using an already-initialized model.
    Accepts path, bytes, or a 1D float32 numpy array.
    """
    with prepare_audio_input(audio_input, decode_wav_bytes=decode_wav_bytes) as (audio, _temp):
        try:
            segments, info = model.transcribe(
                audio,
                language=language,
                task=task,
                vad_filter=vad_filter,
                beam_size=beam_size,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}") from e

    # Collect results
    segment_texts: List[str] = []
    segments_list: List[Dict[str, Any]] = []
    for segment in segments:
        text = segment.text.strip()
        segment_texts.append(text)
        if return_meta:
            segments_list.append(
                {"start": segment.start, "end": segment.end, "text": text}
            )

    full_text = " ".join(segment_texts).strip()
    if not return_meta:
        return full_text

    return {
        "text": full_text,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration_seconds": getattr(info, "duration", None),
        "segments": segments_list,
    }
