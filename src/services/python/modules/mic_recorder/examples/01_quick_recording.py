import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.mic_recorder import MicRecorder

mic_name = 'HyperX SoloCast'

recorder = MicRecorder()
print("Available audio input devices:")
devices = recorder.list_devices()
print(devices)
recorder.set_device(mic_name)

recorder.record_for_duration(5, "C:\\Github\\tft-ai-agent-mastra\\src\\services\\python\\test_1.wav")