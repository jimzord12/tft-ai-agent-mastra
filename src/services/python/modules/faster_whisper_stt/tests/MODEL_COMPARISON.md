# Model Comparison Test Results

This document summarizes the performance of different Whisper models on accuracy and latency.

## Quick Start

Run the comprehensive model comparison tests:

```bash
cd src/services/python

# Compare all models on English audio
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py::TestModelComparison::test_all_models_comparison_english -v -s

# Compare all models on Greek + English audio
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py::TestModelComparison::test_all_models_comparison_greek -v -s

# Run latency benchmarks
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py::TestLatencyBenchmark::test_latency_benchmark_all_models -v -s

# Run all model comparison tests
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py -v -s
```

## Test Results Summary

### English Audio Performance

| Model    | Accuracy | WER    | CER    | Latency | RTF   | Notes                        |
| -------- | -------- | ------ | ------ | ------- | ----- | ---------------------------- |
| tiny     | 80.00%   | 20.00% | 7.84%  | 0.39s   | 0.04x | Fast but less accurate       |
| base     | 100.00%  | 0.00%  | 0.00%  | 0.61s   | 0.06x | **Best balance for English** |
| small    | 100.00%  | 0.00%  | 0.00%  | 1.81s   | 0.18x | High accuracy, still fast    |
| medium   | 80.00%   | 20.00% | 16.67% | 5.13s   | 0.52x | Inconsistent on this sample  |
| large    | 100.00%  | 0.00%  | 0.00%  | 9.05s   | 0.91x | Highest accuracy             |
| large-v2 | 95.00%   | 5.00%  | 3.92%  | 9.22s   | 0.93x | Good but slower              |
| large-v3 | 100.00%  | 0.00%  | 0.00%  | 9.13s   | 0.92x | Latest, most accurate        |

### Greek + English Code-Switching Performance

| Model    | Accuracy | WER    | CER    | Latency | RTF   | How it handles "sexy"    |
| -------- | -------- | ------ | ------ | ------- | ----- | ------------------------ |
| tiny     | 46.15%   | 53.85% | 26.47% | 0.44s   | 0.05x | Mishears as "6"          |
| base     | 69.23%   | 30.77% | 17.65% | 0.71s   | 0.08x | Mishears as "6"          |
| small    | 76.92%   | 23.08% | 7.35%  | 2.12s   | 0.24x | Transliterates as "σέξη" |
| medium   | 100.00%  | 0.00%  | 0.00%  | 5.57s   | 0.62x | **Correctly: "σεξι"**    |
| large    | 100.00%  | 0.00%  | 0.00%  | 10.21s  | 1.14x | Correctly: "σεξι"        |
| large-v2 | 100.00%  | 0.00%  | 0.00%  | 10.21s  | 1.14x | Correctly: "σεξι"        |
| large-v3 | 100.00%  | 0.00%  | 0.00%  | 9.97s   | 1.11x | Correctly: "σεξι"        |

## Key Findings

### For English Audio:

- **Best for production**: `base` model - 100% accuracy with only 0.61s latency (RTF: 0.06x)
- **Best for accuracy**: `large-v3` - 100% accuracy, latest model
- **Fastest**: `tiny` - Only 0.39s latency but 80% accuracy

### For Greek + English Code-Switching:

- **Minimum recommended**: `medium` model for 100% accuracy
- **Code-switching behavior**:
  - Small models (tiny, base) completely mishear English words
  - Medium+ models correctly transliterate English loanwords to Greek
  - English word "sexy" → Greek "σεξι" (proper transliteration)

### Realtime Factor (RTF):

- **< 1.0x**: Faster than realtime (can process audio faster than playback)
- All models achieve RTF < 1.5x on this hardware
- `tiny` and `base` are extremely fast (< 0.1x RTF)

## Recommendations

### Use Case: English-only transcription

✅ **Recommended**: `base` model

- Perfect accuracy for clear English speech
- Very fast (0.06x RTF)
- Good balance of speed and accuracy

### Use Case: Greek with occasional English words

✅ **Recommended**: `medium` or `large-v3` model

- 100% accuracy with proper transliteration
- Correctly handles code-switching
- `medium` is faster (0.62x RTF), `large-v3` slightly more robust

### Use Case: Real-time applications

✅ **Recommended**: `base` model

- Fast enough for real-time processing
- Good accuracy for single languages
- Note: For Greek+English, consider `medium` if accuracy is critical

### Use Case: Batch processing (accuracy critical)

✅ **Recommended**: `large-v3` model

- Highest accuracy across all languages
- Latest improvements from OpenAI
- Worth the extra processing time for critical applications

## Implementation Example

```python
from modules.faster_whisper_stt.transcriber import WhisperSTT

# For English (fast and accurate)
stt = WhisperSTT(model_name="base", device="auto")
result = stt.transcribe("audio.wav", language="en")

# For Greek + English code-switching (accurate)
stt = WhisperSTT(model_name="medium", device="auto")
result = stt.transcribe("audio.wav", language="el", return_meta=True)

# For highest accuracy (any language)
stt = WhisperSTT(model_name="large-v3", device="auto")
result = stt.transcribe("audio.wav", language="el", beam_size=10)
```

## About Code-Switching

**What is code-switching?** Mixing two languages in speech (e.g., Greek with English words).

**How Whisper handles it:**

- Detects the primary language (Greek)
- Transliterates foreign words to match the primary language's script
- Example: English "sexy" → Greek "σεξι"

**Why this happens:**

- The model is trained to output consistent scripts
- Transliteration is often the correct approach for loanwords
- Greek commonly uses transliterated English words in everyday speech

**Important**: Always specify the primary language with `language="el"` for code-switching scenarios. Auto-detection may fail.

## Test File Information

- **English test**: `test_indefinite.wav` - 9.96s duration
- **Greek test**: `test_indefinite_true_02.wav` - 8.94s duration
- Both tests use `beam_size=5` (default)
- All tests use `vad_filter=True` (skip silence)

## Running Individual Tests

```bash
# Test specific model on Greek
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py::TestModelComparison::test_model_accuracy_greek[medium] -v -s

# Test specific model on English
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py::TestModelComparison::test_model_accuracy_english[base] -v -s

# Test latency for fast models
uv run pytest modules/faster_whisper_stt/tests/test_model_comparison.py::TestModelComparison::test_fast_models_latency -v -s
```

## Metrics Explained

- **Accuracy**: Percentage of correctly transcribed words (100 - WER)
- **WER (Word Error Rate)**: Percentage of word-level errors (insertions, deletions, substitutions)
- **CER (Character Error Rate)**: Percentage of character-level errors
- **Latency**: Total processing time in seconds
- **RTF (Realtime Factor)**: Latency / Audio Duration
  - RTF < 1.0 means faster than realtime
  - RTF = 1.0 means same speed as audio
  - RTF > 1.0 means slower than realtime
