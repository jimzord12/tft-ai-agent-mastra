# Mic Recorder Module

A Python module for recording audio from microphone with thread-safe operations.

## Features

- ✅ Simple API for audio recording
- ✅ Device selection by name or index
- ✅ Thread-safe recording operations
- ✅ Support for timed and manual recordings
- ✅ WAV file output and raw PCM data access
- ✅ Real-time audio callback support

## Installation

This module is managed with `uv`. From the project root (`src/services/python`):

```bash
uv sync
```

### System Dependencies

On some systems, you may need to install PortAudio:

**Ubuntu/Debian:**

```bashv
sudo apt-get install portaudio19-dev
```

**macOS:**

```bash
brew install portaudio
```

**Windows:**
PyAudio wheels are usually available. If you encounter issues, install Microsoft Visual C++ Build Tools.

## Usage

### Basic Recording (Timed)

```python
from modules.mic_recorder import MicRecorder

# Create recorder and record for 5 seconds
recorder = MicRecorder()
recorder.record_for_duration(5.0, "output.wav")
```

### List and Select Audio Devices

```python
# List all available input devices
recorder = MicRecorder()
devices = recorder.list_devices()
# Returns: [(0, "Device Name 1"), (1, "Device Name 2"), ...]

# Select device by index
recorder.set_device(1)

# Or select by name (case-insensitive partial match)
recorder.set_device("HyperX SoloCast")
```

### Manual Recording Control

```python
import time

recorder = MicRecorder()
recorder.set_device("HyperX")

# Start recording
recorder.start_recording()

# Record for custom duration
time.sleep(10)

# Stop and get audio data
audio_data = recorder.stop_recording()

# Save to WAV file
recorder.save_to_wav("output.wav", audio_data)
```

### Save Raw PCM Data

```python
from pathlib import Path

# Get raw PCM bytes
audio_data = recorder.stop_recording()

# Save as raw PCM (no WAV headers)
Path("audio.raw").write_bytes(audio_data)

# Can be imported into Audacity or ffmpeg as:
# 44100 Hz, 1 channel (mono), 16-bit signed PCM
```

### Custom Recording Callback

```python
def process_audio_chunk(data: bytes):
    """Process each audio chunk in real-time"""
    print(f"Received {len(data)} bytes")
    # Process audio data here...

recorder = MicRecorder()
recorder.start_recording(callback=process_audio_chunk)
time.sleep(5)
recorder.stop_recording()
```

### Custom Recording Parameters

```python
import pyaudio

recorder = MicRecorder(
    rate=16000,              # 16kHz sample rate
    channels=2,              # Stereo
    format=pyaudio.paInt16,  # 16-bit audio
    chunk=2048,              # Larger buffer
    device_index=1           # Specific device
)
```

## Examples

Check out the `examples/` directory:

- `01_quick_recording.py` - Simple timed recording with device selection
- `02_indefinite_recording.py` - Manual start/stop recording

Run examples with:

```bash
uv run python modules/mic_recorder/examples/01_quick_recording.py
```

## API Reference

### `MicRecorder(rate=44100, channels=1, format=pyaudio.paInt16, chunk=1024, device_index=None)`

**Parameters:**

- `rate`: Sample rate in Hz (default: 44100)
- `channels`: Number of audio channels (1=mono, 2=stereo)
- `format`: Audio format (default: 16-bit PCM)
- `chunk`: Frames per buffer (default: 1024)
- `device_index`: Specific input device index (None = default device)

### Methods

**`list_devices() -> List[Tuple[int, str]]`**

- Returns list of available input devices as (index, name) tuples

**`set_device(device: int | str) -> bool`**

- Set input device by index (int) or name (str)
- String matching is case-insensitive and partial
- Raises `ValueError` if device not found

**`start_recording(callback: Optional[Callable[[bytes], None]] = None)`**

- Start recording audio
- Optional callback receives audio chunks in real-time

**`stop_recording() -> bytes`**

- Stop recording and return raw PCM audio data

**`save_to_wav(filename: str, audio_data: Optional[bytes] = None)`**

- Save audio data to WAV file
- If audio_data is None, uses internally stored frames

**`record_for_duration(duration_sec: float, filename: Optional[str] = None) -> bytes`**

- Convenience method to record for fixed duration
- Returns raw audio data and optionally saves to file

## Testing

Run the unit tests from the project root:

```bash
# From src/services/python
uv run python -m unittest discover -s modules/mic_recorder -p 'test_*.py'

# Or use the script helper
. ./scripts.ps1 test
```

## License

Part of the tft-ai-agent-mastra project.
