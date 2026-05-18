#!/usr/bin/env python3
"""
Checks whether model outputs contain only translations or include explanations/extra text using rule-based approach.
Also detects translations in wrong languages.
"""

import csv
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import unicodedata
from collections import Counter
import warnings
import numpy as np

# Try to import fasttext for language detection
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
    # Suppress fasttext warnings
    fasttext.FastText.eprint = lambda x: None
except ImportError:
    FASTTEXT_AVAILABLE = False
    warnings.warn("fasttext not available. Language detection will use fallback script-based method.")


# Patterns that indicate explanation/extra text (not just translation)
EXPLANATION_PATTERNS = [
    # Common explanation phrases
    r'\b(the translation is|here is|here\'s|this translates to|translation:|translated text:|output:|target:)\b',
    r'\b(in \w+ (this|it) (means|says|translates))\b',
    r'\b(note that|please note|it should be noted)\b',
    r'\b(explanation|reasoning|analysis|breakdown)\b',

    # Meta-linguistic markers
    r'\b(literally|figuratively|idiomatically|contextually)\b',
    r'\b(this (word|phrase|sentence|text))\b',
    r'\b(means|refers to|indicates|suggests)\b.*\b(that|which)\b',

    # Instructions/comments to user
    r'\b(hope this helps|let me know|feel free|if you|you can)\b',
    r'\b(please|kindly|note:|important:)\b',

    # Thinking/reasoning markers
    r'<think>|</think>|<thought>|</thought>',
    r'\*\*reasoning\*\*|\*\*analysis\*\*|\*\*explanation\*\*',

    # Multiple sections with labels
    r'\n\s*(translation|explanation|note|original|source|target)\s*:',

    # Markdown headers
    r'^#+\s+',

    # Numbered or bulleted lists (often explanations)
    r'^\s*[\d\-\*]+[\.\)]\s+',

    # Parenthetical explanations
    r'\([^)]{50,}\)',  # Long parenthetical (likely explanation)

    # XML-style tags
    r'<[a-zA-Z]+>.*</[a-zA-Z]+>',
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in EXPLANATION_PATTERNS]


# Global fasttext model (loaded once)
_FASTTEXT_MODEL = None


def load_fasttext_model(model_path: str = None) -> Optional[object]:
    """
    Load fastText language identification model.

    Args:
        model_path: Path to the fastText model file. If None, tries default locations.

    Returns:
        Loaded fastText model or None if not available
    """
    global _FASTTEXT_MODEL

    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL

    if not FASTTEXT_AVAILABLE:
        return None

    # Try default locations
    if model_path is None:
        possible_paths = [
            Path(__file__).parent.parent / 'models' / 'lid.176.bin',
            Path('models/lid.176.bin'),
            Path('lid.176.bin'),
        ]

        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        else:
            warnings.warn(
                "fastText model not found. Please download it:\n"
                "  mkdir -p models\n"
                "  curl -L -o models/lid.176.bin "
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
            return None

    try:
        _FASTTEXT_MODEL = fasttext.load_model(str(model_path))
        return _FASTTEXT_MODEL
    except Exception as e:
        warnings.warn(f"Failed to load fastText model: {e}")
        return None


# Language code mappings: fastText uses ISO 639-1/639-3, we need to map to our codes
FASTTEXT_TO_STANDARD = {
    'en': 'en',
    'ru': 'ru',
    'fr': 'fr',
    'ar': 'ar',
    'zh': 'zh',
    'ta': 'ta',
    'ko': 'ko',
    'he': 'he',
    'iw': 'he',  # fastText sometimes uses 'iw' for Hebrew
    'it': 'it',
    'de': 'de',
    'cs': 'cs',
    'pl': 'pl',
    'km': 'km',
    'es': 'es',
    'pt': 'pt',
    'ja': 'ja',
    'th': 'th',
    'vi': 'vi',
}


# Language to script mappings
LANG_TO_SCRIPTS = {
    'en': {'LATIN'},
    'ru': {'CYRILLIC'},
    'fr': {'LATIN'},
    'ar': {'ARABIC'},
    'zh': {'HAN', 'CJK'},
    'ta': {'TAMIL'},
    'ko': {'HANGUL', 'HAN', 'CJK'},
    'he': {'HEBREW'},
    'it': {'LATIN'},
    'de': {'LATIN'},
    'cs': {'LATIN'},
    'pl': {'LATIN'},
    'km': {'KHMER'},
}

# Common scripts that are acceptable across languages (punctuation, numbers, etc.)
COMMON_SCRIPTS = {'COMMON', 'INHERITED', 'UNKNOWN'}

# Script ranges for better detection
SCRIPT_RANGES = {
    'ARABIC': range(0x0600, 0x06FF + 1),
    'HEBREW': range(0x0590, 0x05FF + 1),
    'CYRILLIC': range(0x0400, 0x04FF + 1),
    'HAN': list(range(0x4E00, 0x9FFF + 1)) + list(range(0x3400, 0x4DBF + 1)),
    'HANGUL': list(range(0xAC00, 0xD7AF + 1)) + list(range(0x1100, 0x11FF + 1)),
    'TAMIL': range(0x0B80, 0x0BFF + 1),
    'KHMER': range(0x1780, 0x17FF + 1),
    'LATIN': range(0x0000, 0x024F + 1),
    'DEVANAGARI': range(0x0900, 0x097F + 1),
    'THAI': range(0x0E00, 0x0E7F + 1),
}


def get_char_script(char: str) -> str:
    """
    Get the Unicode script of a character.

    Args:
        char: Single character

    Returns:
        Script name (e.g., 'LATIN', 'ARABIC', 'HAN')
    """
    try:
        # Get Unicode script property
        script = unicodedata.name(char, '').split()[0] if unicodedata.name(char, '') else 'UNKNOWN'

        # Map to major script categories
        code_point = ord(char)

        for script_name, ranges in SCRIPT_RANGES.items():
            if code_point in ranges:
                return script_name

        # Fall back to unicode category
        category = unicodedata.category(char)
        if category.startswith('P') or category.startswith('S') or category.startswith('N'):
            return 'COMMON'

        return 'UNKNOWN'
    except:
        return 'UNKNOWN'


def detect_scripts(text: str, min_threshold: float = 0.05) -> Dict[str, float]:
    """
    Detect scripts present in text and their proportions.

    Args:
        text: Input text
        min_threshold: Minimum proportion to include in results

    Returns:
        Dictionary mapping script names to their proportions
    """
    if not text:
        return {}

    # Count characters by script
    script_counts = Counter()
    total_chars = 0

    for char in text:
        if char.strip():  # Ignore whitespace
            script = get_char_script(char)
            script_counts[script] += 1
            total_chars += 1

    if total_chars == 0:
        return {}

    # Calculate proportions
    script_proportions = {}
    for script, count in script_counts.items():
        proportion = count / total_chars
        if proportion >= min_threshold and script not in COMMON_SCRIPTS:
            script_proportions[script] = proportion

    return script_proportions


def detect_language_with_fasttext(text: str, k: int = 1) -> Tuple[Optional[str], float]:
    """
    Detect language using fastText model.

    Args:
        text: Text to detect language for
        k: Number of top predictions to return

    Returns:
        Tuple of (detected_language_code, confidence_score)
        Returns (None, 0.0) if detection fails or model unavailable
    """
    model = load_fasttext_model()

    if model is None or not text or not text.strip():
        return None, 0.0

    try:
        # Clean text: fastText works better with single-line text
        clean_text = ' '.join(text.split())

        # Predict language
        # Note: Need to handle numpy array conversion properly for numpy 2.x
        predictions = model.predict(clean_text, k=k)

        if predictions and len(predictions) == 2:
            labels, scores = predictions

            if len(labels) > 0 and len(scores) > 0:
                # Extract language code (fastText returns '__label__en' format)
                detected_lang = labels[0].replace('__label__', '')

                # Map to standard code
                detected_lang = FASTTEXT_TO_STANDARD.get(detected_lang, detected_lang)

                # Convert numpy array to float safely
                confidence = float(np.asarray(scores[0]))

                return detected_lang, confidence

    except Exception as e:
        warnings.warn(f"fastText detection failed: {e}")

    return None, 0.0


def is_correct_language(translation: str, target_lang: str, threshold: float = 0.5, use_fasttext: bool = True) -> bool:
    """
    Check if translation is primarily in the target language.

    Args:
        translation: The translation text
        target_lang: Target language code (e.g., 'zh', 'ar', 'he', 'en', 'fr')
        threshold: Minimum confidence for language detection (default: 0.5)
        use_fasttext: Whether to use fastText (True) or fallback to script-based detection (False)

    Returns:
        True if translation is in correct language, False otherwise
    """
    if not translation or not target_lang:
        return False

    # Try fastText-based detection first
    if use_fasttext and FASTTEXT_AVAILABLE:
        detected_lang, confidence = detect_language_with_fasttext(translation)

        if detected_lang is not None:
            # For very short texts (< 20 chars), require higher confidence
            min_confidence = 0.7 if len(translation.strip()) < 20 else threshold

            # Check if detected language matches target
            is_match = detected_lang == target_lang

            # If high confidence, trust the result
            if confidence >= min_confidence:
                return is_match

            # For medium confidence (0.3-threshold), be more lenient
            # Could be mixed content or code-switching
            if confidence >= 0.3:
                # Also check scripts as additional signal
                expected_scripts = LANG_TO_SCRIPTS.get(target_lang, set())
                if expected_scripts:
                    script_proportions = detect_scripts(translation, min_threshold=0.05)
                    correct_script_prop = sum(
                        prop for script, prop in script_proportions.items()
                        if script in expected_scripts
                    )
                    # If scripts match well, accept even if language detection uncertain
                    if correct_script_prop >= 0.6:
                        return True

                return is_match

            # Low confidence - fall through to script-based detection

    # Fallback: Script-based detection (original logic)
    # Get expected scripts for target language
    expected_scripts = LANG_TO_SCRIPTS.get(target_lang, set())
    if not expected_scripts:
        # Unknown language, can't check
        return True

    # Detect scripts in translation
    script_proportions = detect_scripts(translation, min_threshold=0.03)

    if not script_proportions:
        # No significant scripts detected (maybe all punctuation?)
        return len(translation.strip()) < 10  # Only accept if very short

    # Calculate proportion in expected scripts
    correct_proportion = sum(
        prop for script, prop in script_proportions.items()
        if script in expected_scripts or any(script in exp for exp in expected_scripts)
    )

    # Check if enough text is in the correct script
    if correct_proportion < 0.6:
        return False

    # Check for excessive mixing (too many different scripts)
    # Allow LATIN for European languages that might have borrowed words
    non_common_scripts = [s for s in script_proportions.keys() if s not in COMMON_SCRIPTS]

    # If target is LATIN-based, allow up to 2 scripts
    if 'LATIN' in expected_scripts:
        max_scripts = 2
    else:
        # For non-Latin scripts, be stricter
        max_scripts = 2  # Allow target script + maybe one other (e.g., numbers, borrowed words)

    if len(non_common_scripts) > max_scripts:
        # Too many different scripts - likely mixed/corrupted
        return False

    return True


def extract_target_lang_from_file_path(file_path: Path) -> str:
    """
    Extract target language from file path (e.g., English-Chinese -> zh).

    Args:
        file_path: Path to translation file

    Returns:
        Target language code
    """
    # Look for language pair in path
    lang_pair = None
    for part in file_path.parts:
        if '-' in part and any(lang in part for lang in LANG_TO_SCRIPTS.keys()):
            lang_pair = part
            break

    if not lang_pair:
        return ''

    # Parse language pair (e.g., "English-Chinese")
    parts = lang_pair.split('-')
    if len(parts) != 2:
        return ''

    # Map language names to codes
    lang_name_to_code = {
        'English': 'en', 'Russian': 'ru', 'French': 'fr', 'Arabic': 'ar',
        'Chinese': 'zh', 'Tamil': 'ta', 'Korean': 'ko', 'Hebrew': 'he',
        'Italian': 'it', 'German': 'de', 'Czech': 'cs', 'Polish': 'pl',
        'Khmer': 'km'
    }

    target_lang_name = parts[1]
    return lang_name_to_code.get(target_lang_name, '')


def contains_explanation(translation: str, reference: str = None) -> bool:
    """
    Check if translation contains explanations or extra text beyond the translation itself.

    Args:
        translation: The model's translation output
        reference: The reference translation (optional, for comparison)

    Returns:
        True if translation contains explanations/extra text, False if it's clean
    """
    if not translation or not translation.strip():
        return True  # Empty translations are not clean

    translation_clean = translation.strip()

    # Check for explicit explanation patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(translation_clean):
            return True

    # Check for multiple paragraphs (likely explanation + translation)
    paragraphs = [p.strip() for p in translation_clean.split('\n\n') if p.strip()]
    if len(paragraphs) > 2:  # More than 2 paragraphs suggests explanation
        return True

    # Check for excessive newlines (formatting suggests structure beyond simple translation)
    newline_count = translation_clean.count('\n')
    if newline_count > 5:  # Arbitrary threshold
        return True

    # Check for quotes around the entire translation (suggests meta-commentary)
    if translation_clean.startswith(('"', "'", '"', '"')) and \
       translation_clean.endswith(('"', "'", '"', '"')) and \
       len(translation_clean) > 100:
        # Check if there's additional text outside quotes
        if translation_clean.count('"') > 2 or translation_clean.count("'") > 2:
            return True

    # Check for colons suggesting labels
    if ':' in translation_clean:
        lines = translation_clean.split('\n')
        for line in lines:
            if ':' in line and len(line.split(':')[0].strip().split()) <= 3:
                # Short text before colon suggests a label
                return True

    # If reference is provided, check length ratio
    if reference:
        ref_len = len(reference.strip())
        trans_len = len(translation_clean)

        # If translation is more than 2x the reference length, likely has explanation
        if trans_len > ref_len * 2 and ref_len > 50:
            return True

    return False


def classify_translation(translation: str, reference: str = None, target_lang: str = None) -> Dict:
    """
    Classify translation as clean (1) or containing explanations/wrong language (0).

    Args:
        translation: The model's translation output
        reference: The reference translation (optional)
        target_lang: Target language code (optional)

    Returns:
        Dictionary containing:
        - label: 1 if clean, 0 if has issues
        - has_explanation: whether explanation was detected
        - wrong_language: whether wrong language was detected
        - detected_lang: detected language code (if available)
        - confidence: detection confidence (if available)
    """
    result = {
        'label': 1,
        'has_explanation': False,
        'wrong_language': False,
        'detected_lang': None,
        'confidence': 0.0
    }

    # Check for explanations
    has_explanation = contains_explanation(translation, reference)
    result['has_explanation'] = has_explanation

    if has_explanation:
        result['label'] = 0

    # Check for correct language if target_lang is provided
    if target_lang:
        # Get language detection info
        detected_lang, confidence = detect_language_with_fasttext(translation)
        result['detected_lang'] = detected_lang
        result['confidence'] = confidence

        # Check if language is correct
        wrong_language = not is_correct_language(translation, target_lang)
        result['wrong_language'] = wrong_language

        if wrong_language:
            result['label'] = 0

    return result


def evaluate_file(file_path: Path, target_lang: str = None) -> Dict:
    """
    Evaluate a single translation file.

    Args:
        file_path: Path to the JSONL translation file
        target_lang: Target language code (extracted from path if not provided)

    Returns:
        Dictionary with evaluation results
    """
    # Extract target language from file path if not provided
    if not target_lang:
        target_lang = extract_target_lang_from_file_path(file_path)

    results = {
        'total': 0,
        'clean': 0,
        'with_issue': 0,  # Renamed from 'with_explanation' to cover both issues
        'wrong_language': 0,
        'has_explanation': 0,
        'classifications': [],
        'target_lang': target_lang
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                translation = entry.get('translation', '')
                reference = entry.get('reference', '')

                # Get target lang from entry if available
                entry_target_lang = entry.get('tgt_lang', target_lang)

                # Overall classification (now returns a dict)
                classification = classify_translation(translation, reference, entry_target_lang)

                label = classification['label']
                has_explanation = classification['has_explanation']
                wrong_language = classification['wrong_language']

                results['total'] += 1
                if label == 1:
                    results['clean'] += 1
                else:
                    results['with_issue'] += 1
                    if has_explanation:
                        results['has_explanation'] += 1
                    if wrong_language:
                        results['wrong_language'] += 1

                results['classifications'].append({
                    'translation': translation,
                    'reference': reference,
                    'label': label,
                    'has_explanation': has_explanation,
                    'wrong_language': wrong_language,
                    'detected_lang': classification.get('detected_lang'),
                    'confidence': classification.get('confidence', 0.0),
                    'target_lang': entry_target_lang
                })

            except json.JSONDecodeError:
                continue

    # Calculate metrics
    if results['total'] > 0:
        results['clean_rate'] = results['clean'] / results['total']
        results['issue_rate'] = results['with_issue'] / results['total']
        results['explanation_rate'] = results['has_explanation'] / results['total']
        results['wrong_lang_rate'] = results['wrong_language'] / results['total']
    else:
        results['clean_rate'] = 0.0
        results['issue_rate'] = 0.0
        results['explanation_rate'] = 0.0
        results['wrong_lang_rate'] = 0.0

    return results


def find_translation_file(output_dir: Path, language_pair: str, split: str = 'test', model_name: str = None) -> Path:
    """
    Find the translation file for a language pair.
    For thinking/reasoning models, only use regular file.
    For other models, prioritizes original file, falls back to regular file.

    Args:
        output_dir: Base output directory for the model
        language_pair: Language pair name
        split: Split name (train/vali/test)
        model_name: Model name to check for thinking/reasoning

    Returns:
        Path to the translation file, or None if not found
    """
    lp_dir = output_dir / language_pair

    # Check if model is a thinking/reasoning model
    is_thinking_model = False
    if model_name:
        model_name_lower = model_name.lower()
        if 'thinking' in model_name_lower or 'reasoning' in model_name_lower or 'r1-distill' in model_name_lower:
            is_thinking_model = True

    # For thinking/reasoning models, only use regular file
    if is_thinking_model:
        regular_file = lp_dir / f"{split}_translations.jsonl"
        if regular_file.exists():
            return regular_file
        return None

    # For other models, try original file first
    original_file = lp_dir / f"{split}_translations_original.jsonl"
    if original_file.exists():
        return original_file

    # Fall back to regular file
    regular_file = lp_dir / f"{split}_translations.jsonl"
    if regular_file.exists():
        return regular_file

    return None


def evaluate_model(model_dir: Path, language_pairs: List[str], split: str = 'test') -> Dict:
    """
    Evaluate all language pairs for a model.

    Args:
        model_dir: Directory containing model outputs
        language_pairs: List of language pairs to evaluate
        split: Split name (train/vali/test)

    Returns:
        Dictionary with evaluation results per language pair
    """
    results = {}
    model_name = model_dir.name

    for lp in language_pairs:
        trans_file = find_translation_file(model_dir, lp, split, model_name)

        if trans_file:
            results[lp] = evaluate_file(trans_file)
            results[lp]['file'] = str(trans_file)
        else:
            results[lp] = {
                'total': 0,
                'clean': 0,
                'with_issue': 0,
                'wrong_language': 0,
                'has_explanation': 0,
                'clean_rate': 0.0,
                'issue_rate': 0.0,
                'explanation_rate': 0.0,
                'wrong_lang_rate': 0.0,
                'file': None,
                'target_lang': ''
            }

    return results


def print_results(all_results: Dict[str, Dict], output_file: str = None):
    """
    Print evaluation results in a readable format.

    Args:
        all_results: Dictionary mapping model names to their results
        output_file: Optional file to write results to
    """
    output_lines = []

    # Header
    line = "=" * 140
    output_lines.append(line)
    output_lines.append("Instruction Following Evaluation Results")
    output_lines.append("Checks for: 1) Explanations/extra text, 2) Wrong language output")
    output_lines.append(line)
    output_lines.append("")

    # Per-model results
    for model_name in sorted(all_results.keys()):
        model_results = all_results[model_name]

        output_lines.append("")
        output_lines.append("=" * 140)
        output_lines.append(f"Model: {model_name}")
        output_lines.append("=" * 140)
        output_lines.append("")
        output_lines.append(
            f"{'Language Pair':<25} {'Total':<8} {'Clean':<8} {'Issues':<8} "
            f"{'w/Expl':<8} {'WrongLng':<9} {'Clean%':<10} {'Expl%':<10} {'WrongL%':<10}"
        )
        output_lines.append("-" * 140)

        # Aggregate metrics
        total_all = 0
        clean_all = 0
        has_expl_all = 0
        wrong_lang_all = 0

        # Sort language pairs
        sorted_lps = sorted(model_results.keys())

        for lp in sorted_lps:
            lp_results = model_results[lp]
            total = lp_results['total']
            clean = lp_results['clean']
            with_issue = lp_results['with_issue']
            has_expl = lp_results['has_explanation']
            wrong_lang = lp_results['wrong_language']
            clean_rate = lp_results['clean_rate']
            expl_rate = lp_results['explanation_rate']
            wrong_lang_rate = lp_results['wrong_lang_rate']

            total_all += total
            clean_all += clean
            has_expl_all += has_expl
            wrong_lang_all += wrong_lang

            output_lines.append(
                f"{lp:<25} {total:<8} {clean:<8} {with_issue:<8} "
                f"{has_expl:<8} {wrong_lang:<9} {clean_rate:<10.2%} "
                f"{expl_rate:<10.2%} {wrong_lang_rate:<10.2%}"
            )

        # Overall metrics for model
        output_lines.append("-" * 140)
        overall_clean_rate = clean_all / total_all if total_all > 0 else 0.0
        overall_expl_rate = has_expl_all / total_all if total_all > 0 else 0.0
        overall_wrong_lang_rate = wrong_lang_all / total_all if total_all > 0 else 0.0
        overall_issue_count = total_all - clean_all
        output_lines.append(
            f"{'OVERALL':<25} {total_all:<8} {clean_all:<8} {overall_issue_count:<8} "
            f"{has_expl_all:<8} {wrong_lang_all:<9} {overall_clean_rate:<10.2%} "
            f"{overall_expl_rate:<10.2%} {overall_wrong_lang_rate:<10.2%}"
        )

    # Comparison across models
    if len(all_results) > 1:
        output_lines.append("")
        output_lines.append("")
        output_lines.append("=" * 140)
        output_lines.append("Model Comparison - Instruction Following Ability")
        output_lines.append("=" * 140)
        output_lines.append("")

        model_overall_stats = []
        for model_name, model_results in all_results.items():
            total = sum(lp_results['total'] for lp_results in model_results.values())
            clean = sum(lp_results['clean'] for lp_results in model_results.values())
            has_expl = sum(lp_results['has_explanation'] for lp_results in model_results.values())
            wrong_lang = sum(lp_results['wrong_language'] for lp_results in model_results.values())

            clean_rate = clean / total if total > 0 else 0.0
            expl_rate = has_expl / total if total > 0 else 0.0
            wrong_lang_rate = wrong_lang / total if total > 0 else 0.0

            model_overall_stats.append((
                model_name, clean_rate, expl_rate, wrong_lang_rate,
                clean, has_expl, wrong_lang, total
            ))

        # Sort by clean rate descending
        model_overall_stats.sort(key=lambda x: x[1], reverse=True)

        output_lines.append(
            f"{'Rank':<6} {'Model':<50} {'Clean%':<12} {'Expl%':<12} {'WrongL%':<12} {'Clean/Total':<15}"
        )
        output_lines.append("-" * 140)

        for rank, (model_name, clean_rate, expl_rate, wrong_lang_rate,
                   clean, has_expl, wrong_lang, total) in enumerate(model_overall_stats, 1):
            output_lines.append(
                f"{rank:<6} {model_name:<50} {clean_rate:<12.2%} {expl_rate:<12.2%} "
                f"{wrong_lang_rate:<12.2%} {clean}/{total}"
            )

    output_lines.append("")
    output_lines.append("=" * 140)

    # Print to console
    for line in output_lines:
        print(line)

    # Write to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\nResults written to: {output_file}")


def save_instance_results(all_results: Dict[str, Dict], output_base: Path, split: str):
    """
    Save per-instance clean/not-clean labels as TSV files.

    For each model and language pair, writes a TSV file next to the translation file
    with columns: index, clean, has_explanation, wrong_language, detected_lang, confidence.

    Args:
        all_results: Dictionary mapping model names to their per-language-pair results
        output_base: Base output directory containing model results
        split: Split name (train/vali/test)
    """
    for model_name, model_results in all_results.items():
        for lp, lp_results in model_results.items():
            classifications = lp_results.get('classifications', [])
            if not classifications:
                continue

            trans_file = lp_results.get('file')
            if not trans_file:
                continue

            # Save TSV alongside the translation file
            tsv_path = Path(trans_file).parent / f"{split}_clean_labels.tsv"

            with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([
                    'index', 'clean', 'has_explanation', 'wrong_language',
                    'detected_lang', 'confidence'
                ])
                for idx, cls in enumerate(classifications):
                    writer.writerow([
                        idx,
                        cls['label'],
                        int(cls['has_explanation']),
                        int(cls['wrong_language']),
                        cls.get('detected_lang', ''),
                        f"{cls.get('confidence', 0.0):.4f}"
                    ])

            print(f"  Saved instance results: {tsv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate instruction-following abilities of LLMs for MT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script checks for two types of instruction-following issues:
  1. Explanations/extra text - The output contains more than just the translation
  2. Wrong language - The output is not in the target language (e.g., Arabic->Hebrew translation contains English/Chinese)

Examples:
  # Evaluate all models in MT_outputs
  python evaluate_instruction_following.py

  # Evaluate specific models
  python evaluate_instruction_following.py --models "Qwen_Qwen3-30B-A3B-Instruct-2507" "meta-llama_Llama-3.2-3B-Instruct"

  # Evaluate from instruction_following_outputs
  python evaluate_instruction_following.py --output-dir instruction_following_outputs/prompt1

  # Save results to file
  python evaluate_instruction_following.py --output instruction_following_results.txt
        """
    )

    parser.add_argument('--output-dir', type=str, default='MT_outputs',
                       help='Base output directory containing model results')
    parser.add_argument('--models', nargs='+', type=str,
                       help='Specific model names to evaluate (default: all models)')
    parser.add_argument('--split', type=str, default='test',
                       help='Split to evaluate (train/vali/test)')
    parser.add_argument('--output', type=str,
                       help='Output file to save results')
    parser.add_argument('--save-instance-results', action='store_true',
                       help='Save per-instance TSV files indicating whether each translation is clean')

    args = parser.parse_args()

    output_base = Path(args.output_dir)

    if not output_base.exists():
        print(f"Error: Output directory not found: {output_base}")
        return

    # Determine which models to evaluate
    if args.models:
        model_dirs = [output_base / model for model in args.models]
        model_dirs = [d for d in model_dirs if d.exists()]

        if not model_dirs:
            print(f"Error: None of the specified models found in {output_base}")
            return
    else:
        # Evaluate all models in the directory
        model_dirs = [d for d in output_base.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"Error: No model directories found in {output_base}")
        return

    # All language pairs
    ALL_LANGUAGE_PAIRS = [
        'Russian-French', 'Arabic-Chinese', 'Chinese-French', 'Chinese-Russian',
        'Arabic-Hebrew', 'Korean-French', 'French-Italian', 'German-French',
        'German-Italian', 'English-Chinese', 'English-Russian', 'English-Tamil',
        'English-German', 'English-Czech', 'English-Polish', 'Chinese-English',
        'Russian-English', 'Tamil-English', 'German-English', 'Czech-English',
        'Khmer-English', 'Korean-Chinese'
    ]

    print(f"\n{'='*140}")
    print(f"Evaluating Instruction Following Abilities")
    print(f"Checks: 1) Explanations/extra text, 2) Wrong language output")
    print(f"{'='*140}")
    print(f"Output directory: {output_base}")
    print(f"Split: {args.split}")
    print(f"Models to evaluate: {len(model_dirs)}")
    for model_dir in sorted(model_dirs):
        print(f"  - {model_dir.name}")
    print(f"{'='*140}\n")

    # Evaluate all models
    all_results = {}

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"Evaluating {model_name}...")

        model_results = evaluate_model(model_dir, ALL_LANGUAGE_PAIRS, args.split)
        all_results[model_name] = model_results

    # Print results
    print_results(all_results, args.output)

    # Save per-instance results if requested
    if args.save_instance_results:
        print("\nSaving per-instance results...")
        save_instance_results(all_results, output_base, args.split)


if __name__ == '__main__':
    main()
