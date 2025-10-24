"""
Unit tests for the WhisperSTT transcriber.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ..transcriber import WhisperSTT


# Test data path
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_AUDIO_FILE = TEST_DATA_DIR / "test_indefinite.wav"


class TestWhisperSTTInitialization:
    """Test WhisperSTT initialization and device selection."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            stt = WhisperSTT()

            assert stt.model_name == "base"
            assert stt.device in ["cpu", "cuda"]
            assert stt.compute_type in ["float16", "float32"]
            mock_model.assert_called_once()

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            stt = WhisperSTT(
                model_name="tiny",
                device="cpu",
                compute_type="float32"
            )

            assert stt.model_name == "tiny"
            assert stt.device == "cpu"
            assert stt.compute_type == "float32"
            mock_model.assert_called_once_with(
                "tiny",
                device="cpu",
                compute_type="float32"
            )

    def test_cuda_available_detection(self):
        """Test CUDA availability detection."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel"):
            # Test when torch is not available
            with patch("modules.faster_whisper_stt.transcriber.WhisperSTT._is_cuda_available", return_value=False):
                stt = WhisperSTT(device="auto", compute_type="auto")
                assert stt.device == "cpu"
                assert stt.compute_type == "float32"

            # Test when CUDA is available
            with patch("modules.faster_whisper_stt.transcriber.WhisperSTT._is_cuda_available", return_value=True):
                stt = WhisperSTT(device="auto", compute_type="auto")
                assert stt.device == "cuda"
                assert stt.compute_type == "float16"


class TestWhisperSTTTranscribeFilePath:
    """Test transcription with file paths."""

    def test_transcribe_valid_file_path_string(self):
        """Test transcription with a valid file path as string."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            # Mock the transcribe method
            mock_segment = MagicMock()
            mock_segment.text = " Test audio "
            mock_segment.start = 0.0
            mock_segment.end = 1.0

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE))

            assert isinstance(result, str)
            assert result == "Test audio"
            mock_instance.transcribe.assert_called_once()

    def test_transcribe_valid_file_path_pathlib(self):
        """Test transcription with a valid file path as Path object."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test "
            mock_segment.start = 0.0
            mock_segment.end = 1.0

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(TEST_AUDIO_FILE)

            assert isinstance(result, str)
            assert result == "Test"

    def test_transcribe_nonexistent_file(self):
        """Test transcription with a nonexistent file path."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel"):
            stt = WhisperSTT()

            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                stt.transcribe("/path/to/nonexistent/file.wav")

    def test_transcribe_with_metadata(self):
        """Test transcription with metadata return."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Hello world "
            mock_segment.start = 0.0
            mock_segment.end = 2.0

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.99
            mock_info.duration = 2.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE), return_meta=True)

            assert isinstance(result, dict)
            assert result["text"] == "Hello world"
            assert result["language"] == "en"
            assert result["language_probability"] == 0.99
            assert result["duration_seconds"] == 2.0
            assert result["model_used"] == "base"
            assert len(result["segments"]) == 1
            assert result["segments"][0]["start"] == 0.0
            assert result["segments"][0]["end"] == 2.0
            assert result["segments"][0]["text"] == "Hello world"


class TestWhisperSTTTranscribeBytes:
    """Test transcription with bytes input."""

    def test_transcribe_bytes_input(self):
        """Test transcription with bytes input."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test bytes "
            mock_segment.start = 0.0
            mock_segment.end = 1.0

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            # Read actual file as bytes
            with open(TEST_AUDIO_FILE, "rb") as f:
                audio_bytes = f.read()

            stt = WhisperSTT()
            result = stt.transcribe(audio_bytes)

            assert isinstance(result, str)
            assert result == "Test bytes"
            mock_instance.transcribe.assert_called_once()

    def test_transcribe_bytes_temp_file_cleanup(self):
        """Test that temporary files are cleaned up after bytes transcription."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test "

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            with open(TEST_AUDIO_FILE, "rb") as f:
                audio_bytes = f.read()

            # Track temp files created
            temp_files = []
            original_named_temp = tempfile.NamedTemporaryFile

            def track_temp_file(*args, **kwargs):
                temp = original_named_temp(*args, **kwargs)
                temp_files.append(temp.name)
                return temp

            with patch("tempfile.NamedTemporaryFile", side_effect=track_temp_file):
                stt = WhisperSTT()
                stt.transcribe(audio_bytes)

            # Check that temp files were cleaned up
            for temp_file in temp_files:
                assert not os.path.exists(temp_file), f"Temp file not cleaned up: {temp_file}"


class TestWhisperSTTTranscribeNumpyArray:
    """Test transcription with numpy array input."""

    def test_transcribe_valid_numpy_array(self):
        """Test transcription with a valid numpy array."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Numpy test "
            mock_segment.start = 0.0
            mock_segment.end = 1.0

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            # Create valid audio array (16kHz mono, 1 second)
            audio_array = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)

            stt = WhisperSTT()
            with pytest.warns(UserWarning, match="Ensure your numpy array is sampled at 16kHz"):
                result = stt.transcribe(audio_array)

            assert isinstance(result, str)
            assert result == "Numpy test"

    def test_transcribe_numpy_array_wrong_dimensions(self):
        """Test that 2D arrays are rejected."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel"):
            stt = WhisperSTT()

            # Create 2D array
            audio_array = np.random.rand(2, 16000).astype(np.float32)

            with pytest.raises(ValueError, match="Audio array must be 1D"):
                stt.transcribe(audio_array)

    def test_transcribe_numpy_array_wrong_dtype(self):
        """Test that non-float32 arrays are converted with warning."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test "

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            # Create int16 array
            audio_array = np.random.randint(-32768, 32767, 16000, dtype=np.int16)

            stt = WhisperSTT()
            with pytest.warns(UserWarning):
                stt.transcribe(audio_array)

    def test_transcribe_numpy_array_out_of_range(self):
        """Test that arrays with values outside [-1.0, 1.0] trigger a warning."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test "

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            # Create array with values outside range
            audio_array = np.random.uniform(-2.0, 2.0, 16000).astype(np.float32)

            stt = WhisperSTT()
            with pytest.warns(UserWarning, match="exceed.*range"):
                stt.transcribe(audio_array)


class TestWhisperSTTTranscribeOptions:
    """Test various transcription options."""

    def test_transcribe_with_language(self):
        """Test transcription with specified language."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Bonjour "

            mock_info = MagicMock()
            mock_info.language = "fr"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE), language="fr")

            # Verify language was passed
            call_args = mock_instance.transcribe.call_args
            assert call_args[1]["language"] == "fr"

    def test_transcribe_with_translate_task(self):
        """Test transcription with translate task."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Hello "

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE), task="translate")

            # Verify task was passed
            call_args = mock_instance.transcribe.call_args
            assert call_args[1]["task"] == "translate"

    def test_transcribe_with_custom_beam_size(self):
        """Test transcription with custom beam size."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test "

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE), beam_size=10)

            # Verify beam_size was passed
            call_args = mock_instance.transcribe.call_args
            assert call_args[1]["beam_size"] == 10

    def test_transcribe_without_vad_filter(self):
        """Test transcription without VAD filter."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_segment = MagicMock()
            mock_segment.text = " Test "

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 1.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([mock_segment], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE), vad_filter=False)

            # Verify vad_filter was passed
            call_args = mock_instance.transcribe.call_args
            assert call_args[1]["vad_filter"] is False


class TestWhisperSTTErrorHandling:
    """Test error handling in transcription."""

    def test_transcribe_invalid_input_type(self):
        """Test that invalid input types raise TypeError."""
        with patch("modules.faster_whisper_stt.transcriber.WhisperModel"):
            stt = WhisperSTT()

            with pytest.raises(TypeError, match="audio_input must be"):
                stt.transcribe(12345)  # Invalid type

    def test_transcribe_model_error(self):
        """Test that model errors are properly wrapped."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            mock_instance = mock_model.return_value
            mock_instance.transcribe.side_effect = Exception("Model failed")

            stt = WhisperSTT()

            with pytest.raises(RuntimeError, match="Transcription failed"):
                stt.transcribe(str(TEST_AUDIO_FILE))


class TestWhisperSTTMultipleSegments:
    """Test transcription with multiple segments."""

    def test_transcribe_multiple_segments(self):
        """Test transcription that returns multiple segments."""
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

        with patch("modules.faster_whisper_stt.transcriber.WhisperModel") as mock_model:
            # Create multiple segments
            seg1 = MagicMock()
            seg1.text = " Hello "
            seg1.start = 0.0
            seg1.end = 1.0

            seg2 = MagicMock()
            seg2.text = " world "
            seg2.start = 1.0
            seg2.end = 2.0

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            mock_info.duration = 2.0

            mock_instance = mock_model.return_value
            mock_instance.transcribe.return_value = ([seg1, seg2], mock_info)

            stt = WhisperSTT()
            result = stt.transcribe(str(TEST_AUDIO_FILE), return_meta=True)

            assert result["text"] == "Hello world"
            assert len(result["segments"]) == 2
            assert result["segments"][0]["text"] == "Hello"
            assert result["segments"][1]["text"] == "world"
