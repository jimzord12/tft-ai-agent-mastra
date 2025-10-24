"""
Comprehensive model comparison tests for WhisperSTT.

This test suite evaluates all available Whisper models (tiny to large-v3) across:
- Transcription accuracy (WER, CER)
- Processing latency
- Language detection confidence
- Model size vs accuracy tradeoffs

Results are presented in a tabular format for easy comparison.
"""
import time
import pytest
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..transcriber import WhisperSTT
from .test_data_mapper import get_test_data, TEST_AUDIO_FILES
from .accuracy_metrics import get_diff_summary


@dataclass
class ModelResult:
    """Results for a single model's transcription."""
    model_name: str
    transcription: str
    accuracy: float
    wer: float
    cer: float
    latency_seconds: float
    language_detected: str
    language_confidence: float
    duration_audio: float
    realtime_factor: float  # latency / audio_duration (lower is better)


# All available Whisper models to test
ALL_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

# Fast test - only test essential models
ESSENTIAL_MODELS = ["tiny", "base", "medium", "large"]


class TestModelComparison:
    """Compare all Whisper models on accuracy and latency."""

    @pytest.mark.parametrize("model_name", ESSENTIAL_MODELS)
    def test_model_accuracy_english(self, model_name):
        """Test each model's accuracy on English audio."""
        test_data = get_test_data("test_indefinite.wav")

        if test_data is None or not test_data.file_path.exists():
            pytest.skip("test_indefinite.wav not found")

        # Initialize model
        stt = WhisperSTT(
            model_name=model_name,
            device="auto",
            compute_type="auto"
        )

        # Time the transcription
        start_time = time.perf_counter()
        result = stt.transcribe(
            test_data.file_path,
            language=test_data.language,
            beam_size=5,
            return_meta=True
        )
        latency = time.perf_counter() - start_time

        transcription = result["text"]
        diff = get_diff_summary(test_data.expected_text, transcription)

        # Calculate realtime factor
        rtf = latency / result["duration_seconds"]

        # Print results
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}")
        print(f"Expected:  \"{test_data.expected_text}\"")
        print(f"Got:       \"{transcription}\"")
        print(f"\nMetrics:")
        print(f"  Accuracy:           {diff['accuracy']:.2f}%")
        print(f"  WER:                {diff['wer']:.2f}%")
        print(f"  CER:                {diff['cer']:.2f}%")
        print(f"  Latency:            {latency:.2f}s")
        print(f"  Audio Duration:     {result['duration_seconds']:.2f}s")
        print(f"  Realtime Factor:    {rtf:.2f}x")
        print(f"  Language:           {result['language']} ({result['language_probability']:.1%})")
        print(f"{'=' * 80}\n")

        # Assert reasonable accuracy (lower threshold for tiny/base)
        if model_name in ["tiny"]:
            min_accuracy = 70.0
        elif model_name in ["base"]:
            min_accuracy = 85.0
        else:
            min_accuracy = 95.0

        assert diff['accuracy'] >= min_accuracy, (
            f"{model_name} accuracy {diff['accuracy']:.2f}% below threshold {min_accuracy}%"
        )

    @pytest.mark.parametrize("model_name", ESSENTIAL_MODELS)
    def test_model_accuracy_greek(self, model_name):
        """Test each model's accuracy on Greek audio with English words."""
        test_data = get_test_data("test_indefinite_true_02.wav")

        if test_data is None or not test_data.file_path.exists():
            pytest.skip("test_indefinite_true_02.wav not found")

        # Initialize model
        stt = WhisperSTT(
            model_name=model_name,
            device="auto",
            compute_type="auto"
        )

        # Time the transcription
        start_time = time.perf_counter()
        result = stt.transcribe(
            test_data.file_path,
            language=test_data.language,
            beam_size=5,
            return_meta=True
        )
        latency = time.perf_counter() - start_time

        transcription = result["text"]
        diff = get_diff_summary(test_data.expected_text, transcription)

        # Calculate realtime factor
        rtf = latency / result["duration_seconds"]

        # Print results
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name} (Greek + English)")
        print(f"{'=' * 80}")
        print(f"Expected:  \"{test_data.expected_text}\"")
        print(f"Got:       \"{transcription}\"")
        print(f"\nMetrics:")
        print(f"  Accuracy:           {diff['accuracy']:.2f}%")
        print(f"  WER:                {diff['wer']:.2f}%")
        print(f"  CER:                {diff['cer']:.2f}%")
        print(f"  Latency:            {latency:.2f}s")
        print(f"  Audio Duration:     {result['duration_seconds']:.2f}s")
        print(f"  Realtime Factor:    {rtf:.2f}x")
        print(f"  Language:           {result['language']} ({result['language_probability']:.1%})")
        print(f"{'=' * 80}\n")

        # Assert language detection is correct
        assert result['language'] == 'el', (
            f"Language detection failed: expected 'el', got '{result['language']}'"
        )

        # Lower accuracy thresholds for Greek with code-switching
        if model_name in ["tiny"]:
            min_accuracy = 50.0
        elif model_name in ["base"]:
            min_accuracy = 65.0
        elif model_name in ["small", "medium"]:
            min_accuracy = 85.0
        else:  # large models
            min_accuracy = 90.0

        assert diff['accuracy'] >= min_accuracy, (
            f"{model_name} accuracy {diff['accuracy']:.2f}% below threshold {min_accuracy}%"
        )

    def test_all_models_comparison_english(self):
        """
        Comprehensive comparison of all models on English audio.

        This test provides a complete comparison table showing the tradeoff
        between model size, accuracy, and processing speed.
        """
        test_data = get_test_data("test_indefinite.wav")

        if test_data is None or not test_data.file_path.exists():
            pytest.skip("test_indefinite.wav not found")

        results: List[ModelResult] = []

        print(f"\n{'=' * 100}")
        print(f"COMPREHENSIVE MODEL COMPARISON - ENGLISH AUDIO")
        print(f"Audio: {test_data.filename}")
        print(f"Description: {test_data.description}")
        print(f"{'=' * 100}\n")

        for model_name in ALL_MODELS:
            try:
                # Initialize model
                stt = WhisperSTT(
                    model_name=model_name,
                    device="auto",
                    compute_type="auto"
                )

                # Time the transcription
                start_time = time.perf_counter()
                result = stt.transcribe(
                    test_data.file_path,
                    language=test_data.language,
                    beam_size=5,
                    return_meta=True
                )
                latency = time.perf_counter() - start_time

                transcription = result["text"]
                diff = get_diff_summary(test_data.expected_text, transcription)
                rtf = latency / result["duration_seconds"]

                model_result = ModelResult(
                    model_name=model_name,
                    transcription=transcription,
                    accuracy=diff['accuracy'],
                    wer=diff['wer'],
                    cer=diff['cer'],
                    latency_seconds=latency,
                    language_detected=result['language'],
                    language_confidence=result['language_probability'],
                    duration_audio=result['duration_seconds'],
                    realtime_factor=rtf
                )
                results.append(model_result)

                print(f"✓ {model_name:12s} - Accuracy: {diff['accuracy']:6.2f}% | "
                      f"Latency: {latency:5.2f}s | RTF: {rtf:5.2f}x")

            except Exception as e:
                print(f"✗ {model_name:12s} - Error: {str(e)}")

        # Print comparison table
        print(f"\n{'=' * 100}")
        print(f"DETAILED RESULTS TABLE")
        print(f"{'=' * 100}")
        print(f"{'Model':<12} {'Accuracy':>9} {'WER':>7} {'CER':>7} {'Latency':>9} {'RTF':>7} {'Lang':>6} {'Conf':>6}")
        print(f"{'-' * 100}")

        for r in results:
            print(f"{r.model_name:<12} {r.accuracy:>8.2f}% {r.wer:>6.2f}% {r.cer:>6.2f}% "
                  f"{r.latency_seconds:>8.2f}s {r.realtime_factor:>6.2f}x "
                  f"{r.language_detected:>6} {r.language_confidence:>6.1%}")

        print(f"{'=' * 100}")
        print(f"\nKey Metrics:")
        print(f"  RTF (Realtime Factor): Processing time / Audio duration")
        print(f"                         < 1.0 means faster than realtime")
        print(f"  WER (Word Error Rate): Percentage of word-level errors")
        print(f"  CER (Character Error Rate): Percentage of character-level errors")
        print(f"{'=' * 100}\n")

        # Store results for potential further analysis
        self._comparison_results = results

    def test_all_models_comparison_greek(self):
        """
        Comprehensive comparison of all models on Greek + English audio.

        This test shows how different models handle code-switching
        between Greek and English.
        """
        test_data = get_test_data("test_indefinite_true_02.wav")

        if test_data is None or not test_data.file_path.exists():
            pytest.skip("test_indefinite_true_02.wav not found")

        results: List[ModelResult] = []

        print(f"\n{'=' * 100}")
        print(f"COMPREHENSIVE MODEL COMPARISON - GREEK + ENGLISH AUDIO")
        print(f"Audio: {test_data.filename}")
        print(f"Description: {test_data.description}")
        print(f"{'=' * 100}\n")

        for model_name in ALL_MODELS:
            try:
                # Initialize model
                stt = WhisperSTT(
                    model_name=model_name,
                    device="auto",
                    compute_type="auto"
                )

                # Time the transcription
                start_time = time.perf_counter()
                result = stt.transcribe(
                    test_data.file_path,
                    language=test_data.language,
                    beam_size=5,
                    return_meta=True
                )
                latency = time.perf_counter() - start_time

                transcription = result["text"]
                diff = get_diff_summary(test_data.expected_text, transcription)
                rtf = latency / result["duration_seconds"]

                model_result = ModelResult(
                    model_name=model_name,
                    transcription=transcription,
                    accuracy=diff['accuracy'],
                    wer=diff['wer'],
                    cer=diff['cer'],
                    latency_seconds=latency,
                    language_detected=result['language'],
                    language_confidence=result['language_probability'],
                    duration_audio=result['duration_seconds'],
                    realtime_factor=rtf
                )
                results.append(model_result)

                print(f"✓ {model_name:12s} - Accuracy: {diff['accuracy']:6.2f}% | "
                      f"Latency: {latency:5.2f}s | RTF: {rtf:5.2f}x | "
                      f"Transcription: \"{transcription[:50]}...\"")

            except Exception as e:
                print(f"✗ {model_name:12s} - Error: {str(e)}")

        # Print comparison table
        print(f"\n{'=' * 100}")
        print(f"DETAILED RESULTS TABLE")
        print(f"{'=' * 100}")
        print(f"{'Model':<12} {'Accuracy':>9} {'WER':>7} {'CER':>7} {'Latency':>9} {'RTF':>7} {'Lang':>6} {'Conf':>6}")
        print(f"{'-' * 100}")

        for r in results:
            print(f"{r.model_name:<12} {r.accuracy:>8.2f}% {r.wer:>6.2f}% {r.cer:>6.2f}% "
                  f"{r.latency_seconds:>8.2f}s {r.realtime_factor:>6.2f}x "
                  f"{r.language_detected:>6} {r.language_confidence:>6.1%}")

        print(f"{'=' * 100}")

        # Show how each model handled the English word "sexy"
        print(f"\nCode-Switching Analysis (English word 'sexy' in Greek context):")
        print(f"{'-' * 100}")
        for r in results:
            # Look for variations of "sexy" or "σεξι" in transcription
            if "sexy" in r.transcription.lower() or "σεξι" in r.transcription.lower():
                word_used = "σεξι" if "σεξι" in r.transcription else "sexy"
                print(f"  {r.model_name:12s}: Used '{word_used}' (transliterated)" if word_used == "σεξι"
                      else f"  {r.model_name:12s}: Kept '{word_used}' (original)")
            else:
                # Extract the word that replaced "sexy"
                words = r.transcription.split()
                if len(words) >= 5:
                    suspected_word = words[4]  # Approximate position
                    print(f"  {r.model_name:12s}: Misheard as '{suspected_word}'")

        print(f"{'=' * 100}\n")

        # Store results
        self._comparison_results_greek = results

    @pytest.mark.parametrize("model_name", ["tiny", "base"])
    def test_fast_models_latency(self, model_name):
        """
        Test that fast models (tiny, base) achieve realtime processing.

        These models should have RTF < 1.0 (faster than realtime).
        """
        test_data = get_test_data("test_indefinite.wav")

        if test_data is None or not test_data.file_path.exists():
            pytest.skip("test_indefinite.wav not found")

        stt = WhisperSTT(model_name=model_name, device="auto", compute_type="auto")

        start_time = time.perf_counter()
        result = stt.transcribe(
            test_data.file_path,
            language="en",
            beam_size=5,
            return_meta=True
        )
        latency = time.perf_counter() - start_time

        rtf = latency / result["duration_seconds"]

        print(f"\n{model_name} Model Performance:")
        print(f"  Audio Duration: {result['duration_seconds']:.2f}s")
        print(f"  Processing Time: {latency:.2f}s")
        print(f"  Realtime Factor: {rtf:.2f}x")

        # Tiny and base should be faster than realtime on most hardware
        # Using 2.0 as threshold to be safe for various hardware
        assert rtf < 2.0, (
            f"{model_name} model too slow: RTF {rtf:.2f}x (should be < 2.0x)"
        )


class TestLatencyBenchmark:
    """Dedicated latency benchmarking tests."""

    def test_latency_benchmark_all_models(self):
        """
        Benchmark processing latency for all models.

        This test helps identify the speed/accuracy tradeoff.
        """
        test_data = get_test_data("test_indefinite.wav")

        if test_data is None or not test_data.file_path.exists():
            pytest.skip("test_indefinite.wav not found")

        print(f"\n{'=' * 80}")
        print(f"LATENCY BENCHMARK")
        print(f"Audio Duration: {test_data.duration_seconds:.2f}s")
        print(f"{'=' * 80}")
        print(f"{'Model':<12} {'Latency':>10} {'RTF':>8} {'Status':>12}")
        print(f"{'-' * 80}")

        for model_name in ESSENTIAL_MODELS:
            try:
                stt = WhisperSTT(model_name=model_name, device="auto", compute_type="auto")

                # Warm-up run (model loading time)
                stt.transcribe(test_data.file_path, language="en")

                # Actual benchmark run
                start_time = time.perf_counter()
                result = stt.transcribe(
                    test_data.file_path,
                    language="en",
                    beam_size=5,
                    return_meta=True
                )
                latency = time.perf_counter() - start_time

                rtf = latency / result["duration_seconds"]
                status = "✓ Realtime" if rtf < 1.0 else "✓ Acceptable" if rtf < 2.0 else "⚠ Slow"

                print(f"{model_name:<12} {latency:>9.2f}s {rtf:>7.2f}x {status:>12}")

            except Exception as e:
                print(f"{model_name:<12} {'ERROR':>9} {'-':>7} {'✗ Failed':>12}")

        print(f"{'=' * 80}\n")
