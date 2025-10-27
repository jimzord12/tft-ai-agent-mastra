"""
Audio input normalization utilities.

This module converts supported audio inputs (path, bytes, ndarray) into a
format acceptable by faster-whisper and handles temporary file cleanup.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Tuple, Union

import numpy as np


AudioInput = Union[str, bytes, np.ndarray]


def _linear_resample_mono_float32(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32)
    ratio = float(target_sr) / float(orig_sr)
    new_len = max(1, int(round(waveform.shape[0] * ratio)))
    # simple linear interpolation
    x_old = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(x_new, x_old, waveform.astype(np.float64))
    return resampled.astype(np.float32)


def _decode_wav_bytes_to_array(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Decode WAV PCM bytes to mono float32 in [-1, 1], resampled to target_sr.
    Falls back by raising an exception for non-PCM or unsupported formats.
    """
    import io
    import wave
    with io.BytesIO(audio_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

    # Convert to numpy based on sample width
    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 1:
        dtype = np.uint8
        scale = 128.0
    elif sampwidth == 3:
        # 24-bit packed PCM: convert manually
        a = np.frombuffer(raw, dtype=np.uint8)
        if a.size % 3 != 0:
            raise ValueError("Invalid 24-bit PCM size")
        a = a.reshape(-1, 3)
        # Sign-extend 24-bit to int32
        signed = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int32) << 16))
        mask = signed & 0x800000
        signed = signed - (mask << 1)
        pcm = signed.astype(np.float32) / float(1 << 23)
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)
        mono = pcm.astype(np.float32)
        return _linear_resample_mono_float32(mono, framerate, target_sr)
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    if sampwidth == 1:
        # 8-bit WAV is unsigned, convert to signed then to float
        pcm = (pcm - 128.0) / 128.0
    else:
        pcm = pcm / scale
    mono = np.clip(pcm, -1.0, 1.0).astype(np.float32)
    return _linear_resample_mono_float32(mono, framerate, target_sr)


@contextmanager
def prepare_audio_input(
    audio_input: AudioInput,
    *,
    decode_wav_bytes: bool = True,
) -> Generator[Tuple[Union[str, np.ndarray], str | None], None, None]:
    """
    Yields a tuple (audio, temp_path) where:
      - audio: str path to file or a numpy float32 mono waveform
      - temp_path: temp file created (if any) to be removed by the caller

    The caller is responsible for deleting temp_path if provided.
    """
    temp_file_path = None
    try:
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            yield str(audio_input), None
        elif isinstance(audio_input, bytes):
            if decode_wav_bytes:
                try:
                    arr = _decode_wav_bytes_to_array(audio_input, target_sr=16000)
                    yield arr, None
                except Exception:
                    # Fallback to temp file for non-WAV or unsupported PCM
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_input)
                        temp_file_path = tmp.name
                    yield temp_file_path, temp_file_path
            else:
                # Always temp-file path path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_input)
                    temp_file_path = tmp.name
                yield temp_file_path, temp_file_path
        elif isinstance(audio_input, np.ndarray):
            arr = audio_input
            if arr.ndim != 1:
                raise ValueError("Audio array must be 1D (mono).")
            if arr.dtype != np.float32:
                warnings.warn("Converting audio array to float32.")
                arr = arr.astype(np.float32)
            if np.abs(arr).max() > 1.0:
                warnings.warn(
                    "Audio array values exceed [-1.0, 1.0]. Consider normalizing."
                )
            warnings.warn(
                "Ensure numpy array is sampled at 16kHz mono for best results."
            )
            yield arr, None
        else:
            raise TypeError(
                "audio_input must be path, bytes, or np.ndarray (1D float32 mono)."
            )
    finally:
        # context ensures temp path is deleted if the caller did not
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
