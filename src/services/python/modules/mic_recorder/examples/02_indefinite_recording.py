import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.mic_recorder import MicRecorder
mic_name = 'HyperX SoloCast'
output_path = "C:\\Github\\tft-ai-agent-mastra\\src\\services\\python\\modules\\mic_recorder\\test_data\\test_indefinite.wav"
raw_path = Path("C:/Github/tft-ai-agent-mastra/src/services/python/modules/mic_recorder/test_data/test_indefinite.raw")

recorder = MicRecorder()
recorder.set_device(mic_name)
recorder.start_recording()
time.sleep(20)  # pauses for 20 seconds
audio_data = recorder.stop_recording()
recorder.save_to_wav(output_path, audio_data)
raw_path.write_bytes(audio_data)