"""
Accuracy metrics for transcription evaluation.

Provides utilities to calculate Word Error Rate (WER), Character Error Rate (CER),
and other accuracy metrics for comparing transcriptions.
"""
from typing import List, Tuple
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by:
    - Converting to lowercase
    - Removing extra whitespace
    - Removing punctuation

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits required
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

    Args:
        reference: The correct transcription (ground truth)
        hypothesis: The actual transcription from the model
        normalize: Whether to normalize text before comparison

    Returns:
        WER as a percentage (0-100)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 100.0 if len(hyp_words) > 0 else 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    wer = (distance / len(ref_words)) * 100

    return wer


def character_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (Character Substitutions + Deletions + Insertions) / Total Characters in Reference

    Args:
        reference: The correct transcription (ground truth)
        hypothesis: The actual transcription from the model
        normalize: Whether to normalize text before comparison

    Returns:
        CER as a percentage (0-100)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    if len(reference) == 0:
        return 100.0 if len(hypothesis) > 0 else 0.0

    distance = levenshtein_distance(reference, hypothesis)
    cer = (distance / len(reference)) * 100

    return cer


def accuracy_score(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate accuracy as percentage of correct matches.

    Accuracy = 100 - WER

    Args:
        reference: The correct transcription (ground truth)
        hypothesis: The actual transcription from the model
        normalize: Whether to normalize text before comparison

    Returns:
        Accuracy as a percentage (0-100)
    """
    wer = word_error_rate(reference, hypothesis, normalize)
    return max(0.0, 100.0 - wer)


def get_diff_summary(reference: str, hypothesis: str, normalize: bool = True) -> dict:
    """
    Get a detailed summary of differences between reference and hypothesis.

    Args:
        reference: The correct transcription (ground truth)
        hypothesis: The actual transcription from the model
        normalize: Whether to normalize text before comparison

    Returns:
        Dictionary with accuracy metrics and comparison details
    """
    original_ref = reference
    original_hyp = hypothesis

    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    wer = word_error_rate(original_ref, original_hyp, normalize)
    cer = character_error_rate(original_ref, original_hyp, normalize)
    acc = accuracy_score(original_ref, original_hyp, normalize)

    return {
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "accuracy": round(acc, 2),
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "reference_chars": len(reference),
        "hypothesis_chars": len(hypothesis),
        "exact_match": reference == hypothesis,
        "normalized_reference": reference,
        "normalized_hypothesis": hypothesis,
    }
