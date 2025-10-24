# Faster Whisper STT Examples

This directory contains practical examples demonstrating how to use the WhisperSTT transcriber.

## Examples

### 01_simple_transcribe.py

A comprehensive example showing basic usage of WhisperSTT:

**Features demonstrated:**

- Loading and transcribing a WAV file
- Basic transcription (text only)
- Getting detailed metadata (language, duration, segments)
- Using different beam sizes for accuracy vs. speed tradeoff
- Working with timestamps and segments

**Run it:**

```bash
cd src/services/python
python modules/faster_whisper_stt/examples/01_simple_transcribe.py
```

**What you'll see:**

- Model initialization details
- Three different transcription examples
- Metadata including language detection, confidence, duration
- Segment-level timestamps
- Tips for improving accuracy

## Usage Patterns

### Quick Start

```python
from modules.faster_whisper_stt.transcriber import WhisperSTT

# Initialize
stt = WhisperSTT(model_name="tiny")

# Transcribe
text = stt.transcribe("path/to/audio.wav")
print(text)
```

### With Metadata

```python
result = stt.transcribe("audio.wav", return_meta=True)

print(f"Text: {result['text']}")
print(f"Language: {result['language']}")
print(f"Duration: {result['duration_seconds']}s")

# Access segments
for segment in result['segments']:
    print(f"[{segment['start']}s - {segment['end']}s] {segment['text']}")
```

### Different Input Types

```python
# File path (string or Path)
text = stt.transcribe("audio.wav")
text = stt.transcribe(Path("audio.wav"))

# Bytes
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()
text = stt.transcribe(audio_bytes)

# Numpy array (16kHz, mono, float32, range [-1.0, 1.0])
import numpy as np
audio_array = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
text = stt.transcribe(audio_array)
```

### Advanced Options

```python
# Specify language
text = stt.transcribe("audio.wav", language="es")

# Translate to English
text = stt.transcribe("french_audio.wav", task="translate")

# Higher accuracy (slower)
text = stt.transcribe("audio.wav", beam_size=10)

# Disable VAD filter
text = stt.transcribe("audio.wav", vad_filter=False)
```

## Model Sizes

Choose based on your accuracy/speed requirements:

| Model    | Size   | Speed   | Accuracy | Best For             |
| -------- | ------ | ------- | -------- | -------------------- |
| tiny     | ~75MB  | Fastest | Basic    | Testing, prototypes  |
| base     | ~150MB | Fast    | Good     | General use          |
| small    | ~500MB | Medium  | Better   | Production           |
| medium   | ~1.5GB | Slow    | Great    | High accuracy needed |
| large-v3 | ~3GB   | Slowest | Best     | Maximum accuracy     |

```python
# Initialize with different models
stt_tiny = WhisperSTT(model_name="tiny")
stt_base = WhisperSTT(model_name="base")
stt_large = WhisperSTT(model_name="large-v3")
```

## Performance Tips

1. **Use appropriate model size**: Start with `tiny` or `base` for testing
2. **Adjust beam_size**: Default is 5, increase to 10+ for better accuracy
3. **Enable VAD filter**: Skips silent parts (enabled by default)
4. **GPU acceleration**: Use `device="cuda"` if you have a GPU
5. **Specify language**: If you know the language, specify it to skip detection

## Common Issues

### Audio Format

- WhisperSTT accepts most audio formats (WAV, MP3, M4A, etc.)
- For numpy arrays, ensure 16kHz sample rate, mono, float32, range [-1.0, 1.0]

### Memory

- Large models require significant RAM/VRAM
- Use `compute_type="int8"` for lower memory usage

### Speed

- First run downloads the model (cached for future use)
- GPU is much faster than CPU
- Smaller models are faster but less accurate

## Adding Your Own Examples

When creating new examples:

1. Use the test audio file or add your own to `tests/data/`
2. Include clear comments and docstrings
3. Show both basic and advanced usage
4. Print informative output
5. Handle errors gracefully
6. Follow the naming convention: `##_description.py`
