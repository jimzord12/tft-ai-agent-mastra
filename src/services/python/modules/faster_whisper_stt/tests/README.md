# Faster Whisper STT Tests

This directory contains comprehensive unit tests for the WhisperSTT transcriber module.

## Test Structure

```
tests/
├── __init__.py
├── test_transcriber.py       # Main unit tests (mocked, fast)
├── test_accuracy.py          # Accuracy tests (uses real models, slower)
├── test_data_mapper.py       # Test data registry and metadata
├── accuracy_metrics.py       # WER, CER, and accuracy calculations
├── data/                     # Test audio files
│   └── test_indefinite.wav
└── README.md                 # This file
```

## Running Tests

### Prerequisites

Make sure you have the dev dependencies installed:

```bash
cd src/services/python
uv sync --group dev
```

### Run All Tests

```bash
# From the project root
cd src/services/python
pytest modules/faster_whisper_stt/tests/

# Run only fast unit tests (mocked, no model downloads)
pytest modules/faster_whisper_stt/tests/test_transcriber.py

# Run accuracy tests (downloads models, slower)
pytest modules/faster_whisper_stt/tests/test_accuracy.py -v

# Or with coverage
pytest modules/faster_whisper_stt/tests/ --cov=modules.faster_whisper_stt --cov-report=html
```

### Run Specific Test Classes

```bash
# Test initialization only
pytest modules/faster_whisper_stt/tests/test_transcriber.py::TestWhisperSTTInitialization

# Test file path transcription
pytest modules/faster_whisper_stt/tests/test_transcriber.py::TestWhisperSTTTranscribeFilePath

# Test error handling
pytest modules/faster_whisper_stt/tests/test_transcriber.py::TestWhisperSTTErrorHandling
```

### Run Specific Tests

```bash
pytest modules/faster_whisper_stt/tests/test_transcriber.py::TestWhisperSTTInitialization::test_init_default_params
```

### Run with Verbose Output

```bash
pytest modules/faster_whisper_stt/tests/ -v
```

## Test Coverage

The test suite covers:

### 1. **Initialization** (`TestWhisperSTTInitialization`)

- Default parameters
- Custom parameters
- CUDA detection and device selection
- Auto device/compute type selection

### 2. **File Path Transcription** (`TestWhisperSTTTranscribeFilePath`)

- String file paths
- Path object file paths
- Nonexistent files (error handling)
- Metadata return
- Multiple segments

### 3. **Bytes Input** (`TestWhisperSTTTranscribeBytes`)

- Bytes transcription
- Temporary file cleanup

### 4. **Numpy Array Input** (`TestWhisperSTTTranscribeNumpyArray`)

- Valid float32 mono arrays
- Wrong dimensions (error)
- Wrong dtype (conversion)
- Out of range values (warning)

### 5. **Transcription Options** (`TestWhisperSTTTranscribeOptions`)

- Language specification
- Translate task
- Custom beam size
- VAD filter on/off

### 6. **Error Handling** (`TestWhisperSTTErrorHandling`)

- Invalid input types
- Model errors
- File not found errors

### 7. **Multiple Segments** (`TestWhisperSTTMultipleSegments`)

- Concatenation of multiple segments
- Segment metadata

## Test Data

The test suite uses a real audio file:

- `data/test_indefinite.wav` - Sample WAV file for transcription testing

## Mocking

Most tests use mocking to avoid:

1. Downloading large model files during testing
2. Long transcription times
3. Network dependencies
4. GPU requirements

Tests verify that the correct parameters are passed to the underlying `faster-whisper` library and that responses are properly formatted.

## Writing New Tests

When adding new features to `transcriber.py`, please:

1. Add corresponding tests in the appropriate test class
2. Test both success and failure cases
3. Use mocking for the WhisperModel to keep tests fast
4. Add docstrings explaining what each test validates
5. Ensure new tests follow the existing naming conventions

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

- No GPU required (tests use mocking)
- No large model downloads
- Fast execution (< 1 second for full suite with mocks)
- No external network calls

## Accuracy Testing

### Adding New Test Audio Files

To add a new test case, edit `test_data_mapper.py`:

```python
TEST_AUDIO_FILES: Dict[str, TestData] = {
    "test_indefinite.wav": TestData(...),

    # Add your new test file here:
    "my_test.wav": TestData(
        filename="my_test.wav",
        expected_text="The exact transcription you expect",
        language="en",  # Language code
        duration_seconds=5.0,
        model_size="base",  # Recommended model
        beam_size=5,
        description="Description of the audio content",
        notes="Any special notes or known issues"
    ),
}
```

Then place your `my_test.wav` file in the `tests/data/` directory.

### Running Accuracy Tests

```bash
# Run all accuracy tests
uv run pytest modules/faster_whisper_stt/tests/test_accuracy.py -v -s

# Run for a specific file (parametrized tests)
uv run pytest modules/faster_whisper_stt/tests/test_accuracy.py::TestTranscriptionAccuracy::test_transcription_accuracy -v
```

### Understanding Accuracy Metrics

The tests calculate three key metrics:

1. **Accuracy** - Percentage of correctly transcribed words (100 - WER)
2. **WER (Word Error Rate)** - Percentage of word-level errors
3. **CER (Character Error Rate)** - Percentage of character-level errors

**Example output:**

```
Expected:  "I am the smartest most beautiful man in the world"
Got:       "I am the smartest, most beautiful man in the world."

Metrics:
  Accuracy: 100.00%
  WER:      0.00%
  CER:      0.00%
  Exact:    True (after normalization)
```

### Accuracy Thresholds

- **95% minimum** - Tests fail if accuracy drops below 95%
- Normalization removes punctuation and case differences
- Language detection must match expected language
