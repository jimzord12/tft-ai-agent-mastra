"""
Automated accuracy tests for WhisperSTT transcriber.

These tests use the test_data_mapper to run transcription accuracy tests
on all registered audio files and compare against expected transcriptions.
"""
import pytest
from unittest.mock import patch

from ..transcriber import WhisperSTT
from .test_data_mapper import get_all_test_files, get_test_data, verify_test_files_exist
from .accuracy_metrics import (
    word_error_rate,
    character_error_rate,
    accuracy_score,
    get_diff_summary
)


class TestTranscriptionAccuracy:
    """Test transcription accuracy against ground truth."""

    def test_all_test_files_exist(self):
        """Verify that all registered test files exist on disk."""
        file_status = verify_test_files_exist()
        missing_files = [f for f, exists in file_status.items() if not exists]

        assert len(missing_files) == 0, (
            f"Missing test files: {', '.join(missing_files)}"
        )

    @pytest.mark.parametrize(
        "test_data",
        get_all_test_files(),
        ids=lambda td: f"{td.filename}_{td.model_size}"
    )
    def test_transcription_accuracy(self, test_data):
        """
        Test transcription accuracy for each registered audio file.

        This test will:
        1. Load the audio file
        2. Transcribe it using the recommended model and settings
        3. Compare against expected transcription
        4. Calculate accuracy metrics (WER, CER, Accuracy)
        5. Assert that accuracy meets minimum threshold
        """
        # Skip if file doesn't exist
        if not test_data.file_path.exists():
            pytest.skip(f"Test file not found: {test_data.filename}")

        # Initialize transcriber with recommended settings
        stt = WhisperSTT(
            model_name=test_data.model_size,
            device="auto",
            compute_type="auto"
        )

        # Transcribe
        result = stt.transcribe(
            test_data.file_path,
            language=test_data.language,
            beam_size=test_data.beam_size,
            return_meta=True
        )

        transcription = result["text"]

        # Calculate accuracy metrics
        diff = get_diff_summary(test_data.expected_text, transcription)

        # Print detailed results
        print(f"\n{'=' * 70}")
        print(f"Test: {test_data.filename}")
        print(f"Description: {test_data.description}")
        print(f"Model: {test_data.model_size}, Beam: {test_data.beam_size}")
        print(f"{'=' * 70}")
        print(f"\nExpected:  \"{test_data.expected_text}\"")
        print(f"Got:       \"{transcription}\"")
        print(f"\nNormalized Expected: \"{diff['normalized_reference']}\"")
        print(f"Normalized Got:      \"{diff['normalized_hypothesis']}\"")
        print(f"\nMetrics:")
        print(f"  Accuracy: {diff['accuracy']:.2f}%")
        print(f"  WER:      {diff['wer']:.2f}%")
        print(f"  CER:      {diff['cer']:.2f}%")
        print(f"  Exact:    {diff['exact_match']}")
        print(f"\nMetadata:")
        print(f"  Language: {result['language']} ({result['language_probability']:.1%} confidence)")
        print(f"  Duration: {result['duration_seconds']:.2f}s")
        print(f"  Segments: {len(result['segments'])}")

        if test_data.notes:
            print(f"\nNotes: {test_data.notes}")

        print(f"{'=' * 70}\n")

        # Assert minimum accuracy threshold
        # Using 95% as threshold (allows for minor punctuation differences)
        assert diff['accuracy'] >= 95.0, (
            f"Accuracy {diff['accuracy']:.2f}% is below threshold 95%. "
            f"WER: {diff['wer']:.2f}%, CER: {diff['cer']:.2f}%"
        )

        # Verify language detection
        assert result['language'] == test_data.language, (
            f"Language mismatch: expected {test_data.language}, "
            f"got {result['language']}"
        )

    def test_specific_audio_file_indefinite(self):
        """Test the test_indefinite.wav file specifically."""
        test_data = get_test_data("test_indefinite.wav")

        if test_data is None:
            pytest.skip("test_indefinite.wav not registered in test_data_mapper")

        if not test_data.file_path.exists():
            pytest.skip(f"Test file not found: {test_data.filename}")

        stt = WhisperSTT(model_name="base", device="auto")

        transcription = stt.transcribe(
            test_data.file_path,
            language="en",  # Must specify English to avoid detection issues
            beam_size=5
        )

        # Get accuracy
        diff = get_diff_summary(test_data.expected_text, transcription)

        print(f"\nExpected: {test_data.expected_text}")
        print(f"Got:      {transcription}")
        print(f"Accuracy: {diff['accuracy']:.2f}%")
        print(f"WER:      {diff['wer']:.2f}%")

        # Should be 100% accurate with base model and language specified
        assert diff['accuracy'] >= 99.0, (
            f"Base model with language='en' should achieve >99% accuracy. "
            f"Got {diff['accuracy']:.2f}%"
        )


class TestAccuracyMetrics:
    """Test the accuracy metric functions themselves."""

    def test_exact_match(self):
        """Test accuracy with exact match."""
        ref = "Hello world"
        hyp = "Hello world"

        assert word_error_rate(ref, hyp) == 0.0
        assert character_error_rate(ref, hyp) == 0.0
        assert accuracy_score(ref, hyp) == 100.0

    def test_one_word_error(self):
        """Test accuracy with one word substitution."""
        ref = "Hello world"
        hyp = "Hello there"

        wer = word_error_rate(ref, hyp)
        assert wer == 50.0  # 1 error out of 2 words

        acc = accuracy_score(ref, hyp)
        assert acc == 50.0

    def test_case_insensitive(self):
        """Test that comparison is case-insensitive when normalized."""
        ref = "Hello World"
        hyp = "hello world"

        assert word_error_rate(ref, hyp, normalize=True) == 0.0
        assert word_error_rate(ref, hyp, normalize=False) > 0.0

    def test_punctuation_ignored(self):
        """Test that punctuation is ignored when normalized."""
        ref = "Hello, world!"
        hyp = "Hello world"

        assert word_error_rate(ref, hyp, normalize=True) == 0.0

    def test_extra_whitespace_normalized(self):
        """Test that extra whitespace is normalized."""
        ref = "Hello  world"
        hyp = "Hello world"

        assert word_error_rate(ref, hyp, normalize=True) == 0.0

    def test_character_error_rate(self):
        """Test character-level accuracy."""
        ref = "abc"
        hyp = "adc"

        # 1 character different out of 3
        cer = character_error_rate(ref, hyp, normalize=False)
        assert abs(cer - 33.33) < 0.1  # Approximately 33.33%

    def test_diff_summary(self):
        """Test that diff summary contains all expected fields."""
        ref = "The quick brown fox"
        hyp = "The quick brown dog"

        diff = get_diff_summary(ref, hyp)

        assert "wer" in diff
        assert "cer" in diff
        assert "accuracy" in diff
        assert "exact_match" in diff
        assert diff["exact_match"] is False
        assert diff["reference_words"] == 4
        assert diff["hypothesis_words"] == 4
