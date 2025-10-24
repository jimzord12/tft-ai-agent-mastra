import sys
import time
from pathlib import Path
import keyboard

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.mic_recorder import MicRecorder

mic_name = 'HyperX SoloCast'
output_path = "C:\\Github\\tft-ai-agent-mastra\\src\\services\\python\\modules\\mic_recorder\\test_data\\test_indefinite_true_02.wav"
raw_path = Path("C:/Github/tft-ai-agent-mastra/src/services/python/modules/mic_recorder/test_data/test_indefinite_true_02.raw")

# Keyboard keys for control
START_KEY = 'f1'  # Press F1 to start recording
STOP_KEY = 'f2'   # Press F2 to stop recording

recorder = MicRecorder()
recorder.set_device(mic_name)

print(f"Press '{START_KEY.upper()}' to START recording")
print(f"Press '{STOP_KEY.upper()}' to STOP recording")
print("Waiting for input...")

# Wait for start key
keyboard.wait(START_KEY)
print("\n🔴 Recording started!")
recorder.start_recording()

# Wait for stop key
keyboard.wait(STOP_KEY)
print("⏹️  Recording stopped!")
audio_data = recorder.stop_recording()

# Save the recording
recorder.save_to_wav(output_path, audio_data)
raw_path.write_bytes(audio_data)

print(f"\n✅ Recording saved to:")
print(f"   WAV: {output_path}")
print(f"   RAW: {raw_path}")
