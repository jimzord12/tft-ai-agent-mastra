# mic_recorder.py

try:
    import pyaudio
except ImportError as e:
    raise ImportError(
        "PyAudio is not installed. Please install it using 'pip install pyaudio'. "
        "Note: On some systems, you may need to install portaudio first. "
        "On Ubuntu/Debian: sudo apt-get install portaudio19-dev. "
        "On macOS: brew install portaudio."
    ) from e
import wave
import threading
from typing import Optional, Callable
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicRecorder:
    def __init__(
        self,
        rate: int = 44100,
        channels: int = 1,
        format: int = pyaudio.paInt16, # pyright: ignore[reportUndefinedVariable]
        chunk: int = 1024,
        device_index: Optional[int] = None,
    ):
        """
        Initialize the microphone recorder.

        Args:
            rate: Sample rate in Hz (e.g., 44100, 16000)
            channels: Number of audio channels (1 = mono, 2 = stereo)
            format: Audio format (e.g., pyaudio.paInt16)
            chunk: Number of frames per buffer
            device_index: Specific input device index (None = default)
        """
        self.rate = rate
        self.channels = channels
        self.format = format
        self.chunk = chunk
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = deque()
        self._frames_lock = threading.Lock()

    def list_devices(self):
        """List all available audio input devices."""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        devices = []
        for i in range(num_devices):
            device = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device.get('maxInputChannels') > 0:
                devices.append((i, device.get('name')))
                logger.info(f"[Input Device] {i}: {device.get('name')}")
        return devices

    def set_device(self, device: int | str) -> bool:
        """
        Set the input device for recording.

        Args:
            device: Either a device index (int) or device name (str) to search for.
                   If string, performs a case-insensitive partial match.

        Returns:
            True if device was found and set, False otherwise.

        Raises:
            ValueError: If device string is provided but no matching device is found.
        """
        if isinstance(device, int):
            # Validate the device index exists
            devices = self.list_devices()
            if any(idx == device for idx, _ in devices):
                self.device_index = device
                logger.info(f"Device index set to {device}")
                return True
            else:
                raise ValueError(f"Device index {device} not found in available devices")

        elif isinstance(device, str):
            # Search for device by name (case-insensitive partial match)
            devices = self.list_devices()
            device_lower = device.lower()
            matching_device = next(
                (idx for idx, name in devices if device_lower in name.lower()),
                None
            )

            if matching_device is not None:
                self.device_index = matching_device
                logger.info(f"Device set to index {matching_device} (matched '{device}')")
                return True
            else:
                raise ValueError(
                    f"No device found matching '{device}'. "
                    f"Available devices: {[name for _, name in devices]}"
                )
        else:
            raise TypeError(f"device must be int or str, got {type(device).__name__}")

    def start_recording(self, callback: Optional[Callable[[bytes], None]] = None):
        """
        Start recording audio.

        Args:
            callback: Optional function to process each audio chunk in real time.
                      If None, audio is stored in self.frames.
        """
        if self.is_recording:
            logger.warning("Already recording!")
            return

        def audio_callback(in_data, frame_count, time_info, status):
            try:
                logger.debug(f"Audio callback received {len(in_data)} bytes")
                if callback:
                    callback(in_data)
                else:
                    with self._frames_lock:
                        self.frames.append(in_data)
                        logger.debug(f"Frames count: {len(self.frames)}")
                return (None, pyaudio.paContinue)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
                return (None, pyaudio.paComplete)

        try:
            # Always use callback mode for proper audio capture
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk,
                stream_callback=audio_callback,
            )
            self.is_recording = True
            with self._frames_lock:
                self.frames.clear()
            logger.info(f"Recording started with callback: {callback is not None}")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            raise

    def stop_recording(self) -> bytes:
        """Stop recording and return raw audio data."""
        if not self.is_recording:
            logger.warning("Not currently recording.")
            return b""

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        logger.info("Recording stopped.")

        # Combine frames into raw bytes
        with self._frames_lock:
            audio_data = b"".join(self.frames) if self.frames else b""
        return audio_data

    def save_to_wav(self, filename: str, audio_data: Optional[bytes] = None):
        """Save recorded audio to a WAV file."""
        with self._frames_lock:
            data = audio_data or b"".join(self.frames)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(data)
        logger.info(f"Audio saved to {filename}")

    def record_for_duration(self, duration_sec: float, filename: Optional[str] = None) -> bytes:
        """Convenience method to record for a fixed duration."""
        import time
        self.start_recording()
        time.sleep(duration_sec)
        audio = self.stop_recording()
        if filename:
            self.save_to_wav(filename, audio)
        return audio

    def __del__(self):
        """Clean up PyAudio on deletion."""
        try:
            logger.debug("Cleaning up MicRecorder resources")
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
                logger.debug("Stream closed")
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
                logger.debug("PyAudio terminated")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")