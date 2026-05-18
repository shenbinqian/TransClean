#!/usr/bin/env python3
"""
Method: QE-based Span Extraction for clean translation detection.

Uses COMET-Kiwi (reference-free QE) to score all contiguous line spans
of each LLM output and selects the highest-scoring span as the extracted
clean translation. Classification is based on coverage ratio.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from comet import download_model, load_from_checkpoint


MAX_LINES = 20  # Cap per output to bound compute


def generate_spans(lines: List[str]) -> List[Dict]:
    """
    Generate all contiguous spans from a list of non-empty lines.

    Returns list of dicts with keys: text, start, end (inclusive indices).
    """
    spans = []
    n = len(lines)
    for i in range(n):
        for j in range(i, n):
            span_text = "\n".join(lines[i:j + 1]).strip()
            if span_text:
                spans.append({"text": span_text, "start": i, "end": j})
    return spans


def classify_batch(
    sources: List[str],
    translations: List[str],
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    coverage_threshold: float = 0.8,
    gpus: int = 1,
) -> List[Dict]:
    """
    Classify a batch of translations using QE-based span extraction.

    Args:
        sources: Source texts.
        translations: LLM translation outputs (potentially noisy).
        model_name: COMET-Kiwi model identifier.
        coverage_threshold: Min coverage ratio to classify as clean. NOT USED for evaluation.
        gpus: Number of GPUs for COMET inference.

    Returns:
        List of classification dicts matching classify_translation() interface,
        plus extra fields: best_span, coverage, qe_score.
    """
    print(f"Loading QE model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    print(f"Model loaded from: {model_path}")

    # Build all QE data samples across all instances
    qe_data = []          # flat list of {"src": ..., "mt": ...}
    instance_map = []     # (instance_idx, span_idx_within_instance)
    span_registry = []    # per-instance list of spans

    for idx, (src, trans) in enumerate(zip(sources, translations)):
        lines = [l for l in trans.split("\n") if l.strip()]
        lines = lines[:MAX_LINES]

        if len(lines) <= 1:
            # Single-line or empty: score the full output directly
            spans = [{"text": trans.strip(), "start": 0, "end": 0}]
        else:
            spans = generate_spans(lines)

        span_registry.append(spans)
        for si, span in enumerate(spans):
            qe_data.append({"src": src, "mt": span["text"]})
            instance_map.append((idx, si))

    print(f"Scoring {len(qe_data)} candidate spans across {len(sources)} instances...")
    model_output = model.predict(qe_data, batch_size=64, gpus=gpus)
    scores = model_output.scores

    # Assign scores back to spans
    for flat_idx, (inst_idx, span_idx) in enumerate(instance_map):
        span_registry[inst_idx][span_idx]["score"] = scores[flat_idx]

    # Select best span per instance and classify
    results = []
    for idx, (trans, spans) in enumerate(zip(translations, span_registry)):
        best = max(spans, key=lambda s: s["score"])
        full_len = len(trans.strip())
        span_len = len(best["text"])
        coverage = span_len / full_len if full_len > 0 else 1.0
        label = 1 if coverage >= coverage_threshold else 0

        results.append({
            "label": label,
            "has_explanation": label == 0,
            "wrong_language": False,
            "detected_lang": None,
            "confidence": best["score"],
            "best_span": best["text"],
            "coverage": coverage,
            "qe_score": best["score"],
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
        description="QE-based span extraction for clean translation detection"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--model",
        type=str,
        default="Unbabel/wmt22-cometkiwi-da",
        help="COMET-Kiwi model name",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
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

    results = classify_batch(
        sources,
        translations,
        model_name=args.model,
        coverage_threshold=args.coverage_threshold,
        gpus=args.gpus,
    )

    # Merge results with original data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item, res in zip(data, results):
            out = {**item, **res}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
