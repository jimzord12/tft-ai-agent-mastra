import os
import warnings
from pathlib import Path
from typing import Union, Optional, Literal, Dict, Any

import numpy as np
from faster_whisper import WhisperModel


class WhisperSTT:
    """
    Local Speech-to-Text using faster-whisper.
    Optimized for file paths, bytes, and (carefully formatted) numpy arrays.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """
        Initialize the Whisper model.

        Args:
            model_name: e.g., 'tiny', 'base', 'small', 'medium', 'large-v3'
            device: 'cpu', 'cuda', or 'auto' (default: auto-detect)
            compute_type: 'int8', 'float16', 'float32', or 'auto'
        """
        if device == "auto":
            device = "cuda" if self._is_cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "float32"

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        vad_filter: bool = True,
        beam_size: int = 5,
        return_meta: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Transcribe audio to text.

        Args:
            audio_input:
                - str/Path: path to audio file (mp3, wav, m4a, etc.)
                - bytes: raw bytes of a complete audio file (e.g., WAV or MP3 content)
                - np.ndarray: 16kHz mono float32 array in range [-1.0, 1.0] (use with caution)
            language: e.g., 'en', 'es'. If None, auto-detect.
            task: 'transcribe' or 'translate' (to English).
            vad_filter: Skip silent parts (improves accuracy & speed).
            beam_size: Beam size for decoding (default: 5, higher = more accurate but slower).
            return_meta: If True, return dict with segments, language, etc.

        Returns:
            str or dict with transcription and optional metadata.
        """
        temp_file_path = None
        try:
            if isinstance(audio_input, (str, Path)):
                # Pass file path directly
                audio_path = Path(audio_input)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                audio = str(audio_input)
            elif isinstance(audio_input, bytes):
                # Write bytes to a temporary file (required by faster-whisper)
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_input)
                    temp_file_path = tmp.name
                audio = temp_file_path
            elif isinstance(audio_input, np.ndarray):
                # Accept numpy array, but warn about strict requirements
                if audio_input.ndim != 1:
                    raise ValueError("Audio array must be 1D (mono).")
                if audio_input.dtype != np.float32:
                    warnings.warn("Converting audio array to float32.")
                    audio_input = audio_input.astype(np.float32)
                # Validate that audio is in the expected range
                if np.abs(audio_input).max() > 1.0:
                    warnings.warn(
                        "Audio array values exceed [-1.0, 1.0] range. "
                        "This may cause poor transcription quality. "
                        "Consider normalizing your audio data."
                    )
                # Note: We cannot validate the sample rate from the array itself
                # The caller must ensure it's 16kHz mono
                warnings.warn(
                    "Ensure your numpy array is sampled at 16kHz. "
                    "Incorrect sample rates will result in poor transcription."
                )
                # faster-whisper expects [-1.0, 1.0] float32 mono at 16kHz
                audio = audio_input
            else:
                raise TypeError(
                    "audio_input must be str, Path, bytes, or np.ndarray (1D, float32, 16kHz mono)."
                )

            # Run transcription
            try:
                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    vad_filter=vad_filter,
                    beam_size=beam_size,
                )
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {str(e)}") from e

            # Collect results
            segment_texts = []
            segments_list = []
            for segment in segments:
                text = segment.text.strip()
                segment_texts.append(text)
                if return_meta:
                    segments_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": text,
                    })

            full_text = " ".join(segment_texts).strip()

            if not return_meta:
                return full_text

            return {
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration_seconds": info.duration,
                "model_used": self.model_name,
                "segments": segments_list,
            }

        finally:
            # Ensure temporary file is cleaned up
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    # Best-effort cleanup; ignore if file is already gone
                    pass

