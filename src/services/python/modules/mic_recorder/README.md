# Mic Recorder Module

A Python module for recording audio from microphone with thread-safe operations.

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. On some systems, you may need to install system dependencies first:

### Ubuntu/Debian:

```bash
sudo apt-get install portaudio19-dev
```

### macOS:

```bash
brew install portaudio
```

### Windows:

PyAudio wheels are usually available, but if you encounter issues, you may need to install Microsoft Visual C++ Build Tools.

## Usage

```python
from mic_recorder import MicRecorder

# Create a recorder instance
recorder = MicRecorder()

# List available devices
devices = recorder.list_devices()

# Record for 5 seconds
audio_data = recorder.record_for_duration(5.0, "output.wav")

# Or use start/stop recording manually
recorder.start_recording()
# ... do something ...
audio_data = recorder.stop_recording()
recorder.save_to_wav("output.wav", audio_data)
```

## Testing

Run the unit tests:

```bash
python -m unittest test_mic_recorder.py
```
