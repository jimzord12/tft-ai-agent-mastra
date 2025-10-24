#!/usr/bin/env python3
"""
Simple example of using WhisperSTT to transcribe an audio file.

This example demonstrates:
- Loading a WAV file
- Transcribing it to text
- Getting detailed metadata about the transcription
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.faster_whisper_stt.transcriber import WhisperSTT


def main():
    """Main function to demonstrate WhisperSTT usage."""

    # Path to the test audio file
    audio_file = Path(__file__).parent.parent / "tests" / "data" / "test_indefinite.wav"

    # Verify the file exists
    if not audio_file.exists():
        print(f"❌ Error: Test audio file not found at {audio_file}")
        print("Please ensure the test data file exists.")
        return 1

    print("=" * 70)
    print("WhisperSTT - Simple Transcription Example")
    print("=" * 70)
    print(f"\n📁 Audio file: {audio_file.name}")
    print(f"📍 Full path: {audio_file}")

    # Initialize the transcriber
    # Using 'base' model for good balance of speed and accuracy
    print("\n🔧 Initializing WhisperSTT with 'base' model...")
    stt = WhisperSTT(
        model_name="base",  # Good balance of speed and accuracy
        device="auto",      # Auto-detect CPU/CUDA
        compute_type="auto" # Auto-select based on device
    )

    print(f"✅ Model loaded: {stt.model_name}")
    print(f"   Device: {stt.device}")
    print(f"   Compute type: {stt.compute_type}")

    # Example 1: Simple transcription (text only)
    print("\n" + "-" * 70)
    print("Example 1: Basic Transcription (text only)")
    print("-" * 70)

    try:
        text = stt.transcribe(
            audio_file,
            language="en",      # Specify English (use None for auto-detect)
            task="transcribe",  # 'transcribe' or 'translate' (to English)
            vad_filter=True,    # Skip silent parts
            beam_size=5         # Beam size for decoding (higher = more accurate but slower)
        )

        print(f"\n📝 Transcription:\n   \"{text}\"")

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        return 1

    # Example 2: Transcription with metadata
    print("\n" + "-" * 70)
    print("Example 2: Transcription with Metadata")
    print("-" * 70)

    try:
        result = stt.transcribe(
            audio_file,
            language="en",    # Specify English for consistent results
            return_meta=True  # Get detailed metadata
        )

        print(f"\n📝 Transcription: \"{result['text']}\"")
        print(f"\n📊 Metadata:")
        print(f"   Language detected: {result['language']}")
        print(f"   Language probability: {result['language_probability']:.2%}")
        print(f"   Duration: {result['duration_seconds']:.2f} seconds")
        print(f"   Model used: {result['model_used']}")
        print(f"   Number of segments: {len(result['segments'])}")

        if result['segments']:
            print(f"\n⏱️  Segments with timestamps:")
            for i, segment in enumerate(result['segments'], 1):
                print(f"   [{segment['start']:.2f}s - {segment['end']:.2f}s] \"{segment['text']}\"")

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        return 1

    # Example 3: Transcription with custom beam size for better accuracy
    print("\n" + "-" * 70)
    print("Example 3: High-Accuracy Transcription (beam_size=10)")
    print("-" * 70)

    try:
        text_accurate = stt.transcribe(
            audio_file,
            language="en",  # Specify English
            beam_size=10    # Higher beam size for better accuracy (slower)
        )

        print(f"\n📝 High-accuracy transcription:\n   \"{text_accurate}\"")
        print("\n💡 Tip: Higher beam_size values generally produce more accurate")
        print("   results but take longer to process.")

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        return 1

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   - Try different model sizes: 'base', 'small', 'medium', 'large-v3'")
    print("   - Experiment with different audio files")
    print("   - Use task='translate' to translate non-English audio to English")
    print("   - Adjust beam_size to balance speed vs accuracy")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
