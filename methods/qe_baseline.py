#!/usr/bin/env python3
"""
Method: Merged QE Span Extraction + Baseline Rule Cleanup.

Pipeline:
  1. QE span extraction selects the best contiguous span (COMET-Kiwi)
  2. Baseline rules strip formatting artifacts from the span
  3. FastText verifies the cleaned span is in the correct language

Requires: comet, fasttext (same env as qe_span_extraction + baseline)
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

from qe_span_extraction import classify_batch as qe_classify_batch
from baseline import is_correct_language


def strip_code_block(text: str) -> str:
    """Remove markdown code block markers (```)."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        if len(lines) >= 2:
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            stripped = "\n".join(lines[start:end]).strip()
    return stripped


def strip_special_formatting(text: str) -> str:
    """Remove [[...]] or {{...}} wrappers."""
    stripped = text.strip()
    if stripped.startswith("[[") and stripped.endswith("]]"):
        stripped = stripped[2:-2].strip()
    elif stripped.startswith("{{") and stripped.endswith("}}"):
        stripped = stripped[2:-2].strip()
    return stripped


def strip_language_prefix(text: str) -> str:
    """Remove language-name prefix like 'Chinese: ...' or 'Translation: ...'."""
    match = re.match(
        r"^(?:Chinese|Arabic|English|French|German|Russian|Hebrew|Czech|Polish|Tamil|"
        r"Khmer|Korean|Italian|Spanish|Japanese|Portuguese|Turkish|"
        r"Translation|Translated|Here is the translation)\s*:\s*",
        text,
        re.IGNORECASE,
    )
    if match:
        return text[match.end():].strip()
    return text


def strip_extra_punctuation(text: str) -> str:
    """Remove trailing repeated punctuation artifacts (!!!, ???)."""
    match = re.search(r"([!?])\1{2,}$", text)
    if match:
        return text[:match.start()] + match.group(1)
    return text


def clean_formatting(text: str) -> str:
    """Apply all baseline formatting cleanup rules to extracted text."""
    cleaned = text
    cleaned = strip_code_block(cleaned)
    cleaned = strip_special_formatting(cleaned)
    cleaned = strip_language_prefix(cleaned)
    cleaned = strip_extra_punctuation(cleaned)
    return cleaned.strip()


def classify_batch(
    sources: List[str],
    translations: List[str],
    tgt_langs: List[str],
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    coverage_threshold: float = 0.8,
    gpus: int = 1,
) -> List[Dict]:
    """
    Classify translations using QE span extraction + baseline cleanup.

    Args:
        sources: Source texts.
        translations: LLM translation outputs (potentially noisy).
        tgt_langs: Target language codes per instance.
        model_name: COMET-Kiwi model identifier.
        coverage_threshold: Min coverage ratio for clean classification.
        gpus: Number of GPUs for COMET inference.

    Returns:
        List of result dicts with extracted_text, qe_score, coverage,
        language_correct, detected_lang.
    """
    # Step 1: QE span extraction
    qe_results = qe_classify_batch(
        sources, translations,
        model_name=model_name,
        coverage_threshold=coverage_threshold,
        gpus=gpus,
    )

    # Steps 2-3: Baseline cleanup + language verification
    results = []
    for qe_res, trans, tgt_lang in zip(qe_results, translations, tgt_langs):
        best_span_raw = qe_res.get("best_span", trans.strip())

        # Step 2: Apply formatting cleanup
        cleaned = clean_formatting(best_span_raw)

        # Step 3: Language verification
        lang_correct = is_correct_language(cleaned, tgt_lang)

        full_len = len(trans.strip())
        cleaned_len = len(cleaned)
        coverage = cleaned_len / full_len if full_len > 0 else 1.0

        results.append({
            "extracted_text": cleaned,
            "best_span_raw": best_span_raw,
            "qe_score": qe_res.get("qe_score", 0.0),
            "coverage": coverage,
            "language_correct": lang_correct,
            "detected_lang": None,
        })

    return results


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="QE span extraction + baseline cleanup for clean translation detection"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--model", type=str, default="Unbabel/wmt22-cometkiwi-da",
        help="COMET-Kiwi model name",
    )
    parser.add_argument(
        "--coverage-threshold", type=float, default=0.8,
        help="Coverage threshold for clean classification",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} instances from {args.input}")

    sources = [item["source"] for item in data]
    translations = [item["translation"] for item in data]
    tgt_langs = [item["tgt_lang"] for item in data]

    results = classify_batch(
        sources, translations, tgt_langs,
        model_name=args.model,
        coverage_threshold=args.coverage_threshold,
        gpus=args.gpus,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item, res in zip(data, results):
            out = {**item, **res}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
