# test_mic_recorder.py

import unittest
from unittest.mock import patch, MagicMock
from modules.mic_recorder.mic_recorder import MicRecorder
import pyaudio
from collections import deque


class TestMicRecorder(unittest.TestCase):

    def setUp(self):
        # Mock PyAudio instance
        self.mock_pyaudio_instance = MagicMock()
        self.mock_stream = MagicMock()
        self.mock_pyaudio_instance.open.return_value = self.mock_stream

        # Patch PyAudio globally in the module
        self.patcher = patch('modules.mic_recorder.mic_recorder.pyaudio.PyAudio', return_value=self.mock_pyaudio_instance)
        self.MockPyAudio = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_init_defaults(self):
        """Test default initialization."""
        recorder = MicRecorder()
        self.assertEqual(recorder.rate, 44100)
        self.assertEqual(recorder.channels, 1)
        self.assertEqual(recorder.format, pyaudio.paInt16)
        self.assertEqual(recorder.chunk, 1024)
        self.assertIsNone(recorder.device_index)
        self.assertFalse(recorder.is_recording)
        self.assertEqual(recorder.frames, deque())
        self.MockPyAudio.assert_called_once()

    def test_init_custom_args(self):
        """Test initialization with custom arguments."""
        recorder = MicRecorder(rate=16000, channels=2, format=pyaudio.paFloat32, chunk=512, device_index=1)
        self.assertEqual(recorder.rate, 16000)
        self.assertEqual(recorder.channels, 2)
        self.assertEqual(recorder.format, pyaudio.paFloat32)
        self.assertEqual(recorder.chunk, 512)
        self.assertEqual(recorder.device_index, 1)

    @patch('modules.mic_recorder.mic_recorder.logger')
    def test_start_recording_already_recording(self, mock_logger):
        """Test starting recording when already recording logs a warning."""
        recorder = MicRecorder()
        recorder.is_recording = True
        recorder.start_recording()
        mock_logger.warning.assert_called_once_with("Already recording!")

    @patch('modules.mic_recorder.mic_recorder.logger')
    def test_stop_recording_not_recording(self, mock_logger):
        """Test stopping recording when not recording logs a warning."""
        recorder = MicRecorder()
        recorder.is_recording = False
        result = recorder.stop_recording()
        mock_logger.warning.assert_called_with("Not currently recording.")
        self.assertEqual(result, b"")

    def test_start_recording_success(self):
        """Test successful start of recording."""
        recorder = MicRecorder()
        recorder.start_recording()
        # Verify open was called once
        self.mock_pyaudio_instance.open.assert_called_once()
        # Get the actual call arguments
        call_args = self.mock_pyaudio_instance.open.call_args
        # Verify all parameters except stream_callback
        self.assertEqual(call_args.kwargs['format'], recorder.format)
        self.assertEqual(call_args.kwargs['channels'], recorder.channels)
        self.assertEqual(call_args.kwargs['rate'], recorder.rate)
        self.assertEqual(call_args.kwargs['input'], True)
        self.assertEqual(call_args.kwargs['input_device_index'], recorder.device_index)
        self.assertEqual(call_args.kwargs['frames_per_buffer'], recorder.chunk)
        # Verify stream_callback is a callable (implementation always uses callback mode)
        self.assertTrue(callable(call_args.kwargs['stream_callback']))
        # Stream doesn't need to be started explicitly in callback mode
        self.mock_stream.start_stream.assert_not_called()
        self.assertTrue(recorder.is_recording)
        self.assertEqual(recorder.frames, deque())

    def test_start_recording_with_callback(self):
        """Test starting recording with a callback (no explicit start_stream)."""
        recorder = MicRecorder()
        def dummy_callback(data): pass
        recorder.start_recording(callback=dummy_callback)
        # Callback mode doesn't call start_stream
        self.mock_stream.start_stream.assert_not_called()
        self.assertTrue(recorder.is_recording)

    def test_stop_recording_success(self):
        """Test stopping recording and resetting state."""
        recorder = MicRecorder()
        recorder.is_recording = True
        recorder.stream = self.mock_stream
        recorder.frames = deque([b'data1', b'data2'])
        result = recorder.stop_recording()
        self.mock_stream.stop_stream.assert_called_once()
        self.mock_stream.close.assert_called_once()
        self.assertFalse(recorder.is_recording)
        self.assertEqual(result, b'data1data2')

    @patch('modules.mic_recorder.mic_recorder.wave')
    def test_save_to_wav(self, mock_wave):
        """Test saving recorded audio to a WAV file."""
        mock_wf = MagicMock()
        mock_wave.open.return_value.__enter__.return_value = mock_wf
        recorder = MicRecorder()
        self.mock_pyaudio_instance.get_sample_size.return_value = 2
        audio_data = b'raw_audio_data'
        recorder.save_to_wav("output.wav", audio_data)
        mock_wave.open.assert_called_once_with("output.wav", "wb")
        mock_wf.setnchannels.assert_called_once_with(1)
        mock_wf.setsampwidth.assert_called_once()
        mock_wf.setframerate.assert_called_once_with(44100)
        mock_wf.writeframes.assert_called_once_with(audio_data)

    @patch('modules.mic_recorder.mic_recorder.wave')
    @patch('time.sleep')
    def test_record_for_duration(self, mock_sleep, mock_wave):
        """Test the convenience method for fixed-duration recording."""
        mock_wf = MagicMock()
        mock_wave.open.return_value.__enter__.return_value = mock_wf

        recorder = MicRecorder()
        recorder.start_recording = MagicMock()
        recorder.stop_recording = MagicMock(return_value=b'audio_data')
        self.mock_pyaudio_instance.get_sample_size.return_value = 2

        result = recorder.record_for_duration(3.0, "test.wav")
        recorder.start_recording.assert_called_once()
        mock_sleep.assert_called_once_with(3.0)
        recorder.stop_recording.assert_called_once()
        self.assertEqual(result, b'audio_data')

    def test_del_cleans_up(self):
        """Test that __del__ terminates the PyAudio instance."""
        recorder = MicRecorder()
        recorder.__del__()
        self.mock_pyaudio_instance.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()