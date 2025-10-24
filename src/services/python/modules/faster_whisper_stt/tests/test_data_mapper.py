"""
Test data mapper for audio transcription tests.

This module defines the expected transcriptions and metadata for test audio files,
making it easy to add new test cases and run automated accuracy tests.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestData:
    """
    Metadata for a test audio file.

    Attributes:
        filename: Name of the audio file (e.g., "test_indefinite.wav")
        expected_text: The correct transcription (ground truth)
        language: The language code (e.g., "en", "es", "fr")
        duration_seconds: Expected duration of the audio
        model_size: Recommended model size for this test ("tiny", "base", "small", etc.)
        beam_size: Recommended beam size for accuracy
        description: Human-readable description of what the audio contains
        notes: Any additional notes about this test case
    """
    filename: str
    expected_text: str
    language: str
    duration_seconds: float
    model_size: str = "large" # available sizes: tiny, base, small, medium, large, large-v2, large-v3
    beam_size: int = 5
    description: str = ""
    notes: str = ""

    @property
    def file_path(self) -> Path:
        """Get the full path to the test audio file."""
        return Path(__file__).parent / "data" / self.filename


# Test data registry
# Add new test cases here to automatically include them in accuracy tests
TEST_AUDIO_FILES: Dict[str, TestData] = {
    "test_indefinite.wav": TestData(
        filename="test_indefinite.wav",
        expected_text="I am the smartest most beautiful man in the world, did you know that? Who is the smartest, most beautiful.",
        language="en",
        duration_seconds=9.96,
        model_size="large",
        beam_size=5,
        description="Male voice speaking in English about being the smartest and most beautiful",
        notes="Auto-detection fails (detects as Spanish), must specify language='en'. "
              "Tiny model mishears 'the smartest' as 'this artist' or 'this Martin's'. "
              "Base model with language='en' achieves 100% accuracy."
    ),
        "test_indefinite_true_01.wav": TestData(
        filename="test_indefinite_true_01",
        expected_text="Τι λες να κάνω σε αυτόν τον γύρο, να συνεχίσω τα rerolls, να κάνω οικονόμια ή κάτι άλλο;",
        language="el",
        duration_seconds=9.96,
        model_size="large",
        beam_size=5,
        description="Male voice speaking in Greek about TFT game strategies",
        notes="Auto-detection fails (detects as Spanish), must specify language='el'. "
              "Tiny model mishears 'the smartest' as 'this artist' or 'this Martin's'. "
              "Base model with language='el' achieves 100% accuracy."
    ),
        "test_indefinite_true_02.wav": TestData(
        filename="test_indefinite_true_02.wav",
        expected_text="Είμαι ο πιο όμορφος σεξι άντρας στον πλανήτη. Και ο Σταύρος δεν είναι.",
        language="el",
        duration_seconds=8.94,
        model_size="large",
        beam_size=5,
        description="Male voice speaking in Greek with English loanword 'sexy' (transcribed as σεξι)",
        notes="English word 'sexy' is transliterated to Greek as 'σεξι' by larger models. "
              "Auto-detection may fail, must specify language='el'. "
              "Large models achieve 95%+ accuracy with proper transliteration."
    ),
    # Add more test files here:
    # "example_french.wav": TestData(
    #     filename="example_french.wav",
    #     expected_text="Bonjour, comment allez-vous?",
    #     language="fr",
    #     duration_seconds=2.5,
    #     model_size="base",
    #     beam_size=5,
    #     description="French greeting",
    # ),
}


def get_test_data(filename: str) -> Optional[TestData]:
    """
    Get test data for a specific audio file.

    Args:
        filename: Name of the audio file

    Returns:
        TestData object or None if not found
    """
    return TEST_AUDIO_FILES.get(filename)


def get_all_test_files() -> List[TestData]:
    """
    Get all registered test files.

    Returns:
        List of all TestData objects
    """
    return list(TEST_AUDIO_FILES.values())


def add_test_file(test_data: TestData) -> None:
    """
    Register a new test file.

    Args:
        test_data: TestData object to register
    """
    TEST_AUDIO_FILES[test_data.filename] = test_data


def verify_test_files_exist() -> Dict[str, bool]:
    """
    Check which test files actually exist on disk.

    Returns:
        Dictionary mapping filename to existence status
    """
    return {
        filename: data.file_path.exists()
        for filename, data in TEST_AUDIO_FILES.items()
    }
