"""
evaluation_utils.py
===================
Shared labeling logic for all evaluation methods.

Provides:
  - Text normalization and exact-match comparison
  - Detection labeling (extracted == original → clean)
  - Coverage-based labeling (for curated data)
  - Exact-match labeling (for synthetic data)
  - Pattern-to-category mapping
"""

import unicodedata


PATTERN_TO_CATEGORY = {
    "none": "clean",
    "language_prefix": "formatting",
    "verbose_preamble": "formatting",
    "translation_prefix": "formatting",
    "code_block": "formatting",
    "special_formatting": "formatting",
    "extra_punctuation": "formatting",
    "explanation": "content",
    "cultural_note": "content",
    "bilingual_output": "content",
    "alternatives": "content",
    "wrong_language": "content",
    "off_topic": "content",
}

VALID_NOISE_PATTERNS = set(PATTERN_TO_CATEGORY.keys())


def normalize_text(text: str) -> str:
    """Strip leading/trailing whitespace and normalize unicode to NFC form."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


def is_exact_match(extracted: str, reference: str) -> bool:
    """Normalized comparison: normalize_text on both, then ==."""
    return normalize_text(extracted) == normalize_text(reference)


def compute_coverage(extracted: str, original: str) -> float:
    """Compute len(extracted.strip()) / len(original.strip()), handling empty strings."""
    ext_len = len(extracted.strip()) if extracted else 0
    orig_len = len(original.strip()) if original else 0
    if orig_len == 0:
        return 1.0
    return ext_len / orig_len


def assign_label_synthetic(extracted: str, reference: str) -> tuple:
    """
    Exact match with reference -> label 1 (clean) or 0 (noisy).

    Returns:
        (label, {"match": bool, "method": "exact_match"})
    """
    match = is_exact_match(extracted, reference)
    label = 1 if match else 0
    return label, {"match": match, "method": "exact_match"}


def assign_label_curated(
    extracted: str,
    original: str,
    coverage_threshold: float = 0.8,
) -> tuple:
    """
    Coverage-based labeling for curated data.

    coverage > threshold → label 1 (predicts clean)
    coverage <= threshold → label 0 (predicts noisy)

    Returns:
        (label, {"coverage": float, "method": "coverage"})
    """
    coverage = compute_coverage(extracted, original)
    label = 1 if coverage > coverage_threshold else 0
    return label, {"coverage": coverage, "method": "coverage"}


def assign_detection_label(extracted: str, original_translation: str) -> int:
    """
    Detection: did the method detect noise?

    If extracted == original_translation (normalized), the method did not modify
    the text, predicting it is clean (label=1).
    Otherwise, the method changed the text, predicting it is noisy (label=0).
    """
    if is_exact_match(extracted, original_translation):
        return 1
    return 0


def derive_noise_category(noise_pattern: str) -> str:
    """Look up noise category from noise pattern using PATTERN_TO_CATEGORY mapping."""
    if not noise_pattern:
        return None
    return PATTERN_TO_CATEGORY.get(noise_pattern)
