#!/usr/bin/env python3
"""
evaluate_benchmark.py
=====================
Evaluation harness for clean translation detection methods.

Computes two separate metrics:
  1. Detection accuracy: did the method detect whether translation is noisy?
     (extracted == original LLM output → predicts clean, else predicts noisy)
  2. Extraction accuracy: did the method correctly extract the clean translation?
     (exact match with reference for synthetic, silver_reference for curated)

Reads saved results from individual method runs. Does NOT run any methods
directly (each method has its own script due to environment conflicts).

Usage:
    python methods/evaluate_benchmark.py \
        --benchmark data/synthetic.jsonl \
        --results results/llm_extractor/aya/synthetic_results.jsonl \
                  results/qe_span/synthetic_results.jsonl \
        --eval-mode synthetic

    python methods/evaluate_benchmark.py \
        --benchmark data/curated_noisy_1100_with_silver.jsonl \
        --results results/llm_extractor/aya/curated_results.jsonl \
        --eval-mode curated
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_utils import (
    normalize_text,
    is_exact_match,
    assign_detection_label,
    compute_coverage,
    derive_noise_category,
    PATTERN_TO_CATEGORY,
)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def infer_method_name(results_path: str) -> str:
    """Infer method name from results file path.

    e.g. 'results/llm_extractor/qwen3.5_0.8/synthetic_results.jsonl' -> 'llm_extractor/qwen3.5_0.8'
    """
    parts = Path(results_path).parts
    # Find 'results' in path and take everything after it except the filename
    try:
        results_idx = list(parts).index("results")
        return "/".join(parts[results_idx + 1 : -1])
    except ValueError:
        return Path(results_path).stem


def detect_eval_mode(benchmark_data: List[Dict]) -> str:
    """Auto-detect evaluation mode from benchmark data.

    Returns 'synthetic' if data has synthetic noise categories and reference != translation.
    Returns 'curated' otherwise.
    """
    synthetic_categories = {"formatting", "content", "explanation", "combo", "clean"}
    has_synthetic_cats = False
    has_distinct_ref = False

    for item in benchmark_data:
        cat = item.get("noise_category", "")
        if cat in synthetic_categories:
            has_synthetic_cats = True
        ref = item.get("reference", "")
        trans = item.get("translation", "")
        if ref and trans and ref != trans:
            has_distinct_ref = True
        if has_synthetic_cats and has_distinct_ref:
            return "synthetic"

    return "curated"


def _parse_llm_json(text: str) -> Optional[Dict]:
    """Extract and parse JSON from text that may contain markdown code fences.

    Falls back to regex extraction when JSON is malformed (e.g. missing commas,
    unescaped special quotes like „" or «»).
    """
    # Try stripping markdown code fences first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Regex fallback: extract field values from malformed JSON
    result = {}
    # Extract extracted_translation: match the value between the key and noise_pattern
    trans_match = re.search(
        r'"extracted_translation"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,\n]?\s*"noise_pattern',
        text, re.DOTALL,
    )
    if not trans_match:
        # Handle special quotes (e.g. „…", «…») or missing comma before noise_pattern
        trans_match = re.search(
            r'"extracted_translation"\s*:\s*["\u201e\u201c\u00ab]'
            r'(.*?)'
            r'["\u201d\u201c\u00bb]\s*[,\n]?\s*"noise_pattern',
            text, re.DOTALL,
        )
    if not trans_match:
        # Last resort: grab everything between the key and noise_pattern,
        # stripping any leading/trailing quote-like characters
        last_resort = re.search(
            r'"extracted_translation"\s*:\s*(.+?),?\s*\n\s*"noise_pattern',
            text, re.DOTALL,
        )
        if last_resort:
            val = last_resort.group(1).strip().strip('""\u201e\u201c\u201d\u00ab\u00bb')
            result["extracted_translation"] = val.strip()
    else:
        result["extracted_translation"] = trans_match.group(1).strip()
    # Extract noise_pattern
    pattern_match = re.search(
        r'"noise_pattern"\s*:\s*"([^"]*)"', text,
    )
    if pattern_match:
        result["noise_pattern"] = pattern_match.group(1)
    return result if result else None


def get_extracted_text(result: Dict) -> Optional[str]:
    """Get extracted text from a result dict, checking multiple possible field names.

    For LLM extractor results, extracted_text is a JSON string (possibly with
    markdown fences) containing an 'extracted_translation' field — parse it.
    For QE-based results, best_span is a plain string — return as-is.
    """
    if "best_span" in result and result["best_span"] is not None:
        return result["best_span"]

    if "extracted_text" in result and result["extracted_text"] is not None:
        raw = result["extracted_text"]
        parsed = _parse_llm_json(raw)
        if parsed and "extracted_translation" in parsed:
            val = parsed["extracted_translation"]
            if isinstance(val, list):
                return val[0] if val else None
            return str(val) if val is not None else None
        # Fallback: return raw text (handles plain-string outputs)
        return raw

    return None


def compute_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    """Compute precision, recall, F1 for the positive class (label=1 = clean)."""
    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        if p == 1 and g == 1:
            tp += 1
        elif p == 1 and g == 0:
            fp += 1
        elif p == 0 and g == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / len(gold) if len(gold) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_breakdown(
    benchmark: List[Dict],
    gold_labels: List[int],
    pred_labels: List[int],
) -> Dict[str, Dict]:
    """Compute per-category breakdown of metrics."""
    category_field = None
    for field in ("noise_category", "perturbation_type"):
        if any(field in item for item in benchmark):
            category_field = field
            break

    if not category_field:
        return {}

    type_groups = defaultdict(lambda: ([], []))
    for item, g, p in zip(benchmark, gold_labels, pred_labels):
        ptype = item.get(category_field, "unknown")
        type_groups[ptype][0].append(g)
        type_groups[ptype][1].append(p)

    breakdown = {}
    noisy_gold = []
    noisy_pred = []
    for ptype, (g_list, p_list) in type_groups.items():
        breakdown[ptype] = compute_metrics(g_list, p_list)
        if ptype != "clean":
            noisy_gold.extend(g_list)
            noisy_pred.extend(p_list)

    if noisy_gold:
        breakdown["noisy_only"] = compute_metrics(noisy_gold, noisy_pred)

    return breakdown


# ─────────────────────────────────────────────────────────────
# Detection accuracy
# ─────────────────────────────────────────────────────────────

def evaluate_detection(
    benchmark: List[Dict],
    results: List[Dict],
) -> tuple:
    """
    Detection: did the method detect whether translation is noisy?

    If extracted == original LLM output (translation field), the method predicts
    clean (label=1). Otherwise it predicts noisy (label=0).

    Returns:
        (pred_labels, metrics_dict)
    """
    gold_labels = [item.get("label", 0) for item in benchmark]
    pred_labels = []

    for bench_item, res_item in zip(benchmark, results):
        extracted = get_extracted_text(res_item)
        if extracted is None or extracted.strip() == "":
            # No extraction or empty → method found no clean translation → predicts noisy
            pred_labels.append(0)
            continue

        original = bench_item.get("translation", "")
        label = assign_detection_label(extracted, original)
        pred_labels.append(label)

    metrics = compute_metrics(gold_labels, pred_labels)
    return pred_labels, metrics


# ─────────────────────────────────────────────────────────────
# Extraction accuracy
# ─────────────────────────────────────────────────────────────

def compute_extraction_breakdown(
    benchmark: List[Dict],
    match_flags: List[bool],
) -> Dict[str, Dict]:
    """Compute per-category extraction accuracy breakdown."""
    category_field = None
    for field in ("noise_category", "perturbation_type"):
        if any(field in item for item in benchmark):
            category_field = field
            break

    if not category_field:
        return {}

    type_groups = defaultdict(lambda: (0, 0))  # (correct, total)
    noisy_correct = 0
    noisy_total = 0
    for item, matched in zip(benchmark, match_flags):
        ptype = item.get(category_field, "unknown")
        correct, total = type_groups[ptype]
        type_groups[ptype] = (correct + int(matched), total + 1)
        if ptype != "clean":
            noisy_correct += int(matched)
            noisy_total += 1

    breakdown = {}
    for ptype, (correct, total) in type_groups.items():
        breakdown[ptype] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

    if noisy_total > 0:
        breakdown["noisy_only"] = {
            "accuracy": noisy_correct / noisy_total,
            "correct": noisy_correct,
            "total": noisy_total,
        }

    return breakdown


def evaluate_extraction(
    benchmark: List[Dict],
    results: List[Dict],
    eval_mode: str,
) -> Optional[Dict]:
    """
    Extraction accuracy: does the extracted text match the ground truth?

    For synthetic: ground truth = reference field.
    For curated: ground truth = silver_reference field (from silver label aggregation).

    Returns metrics dict or None if ground truth is unavailable.
    """
    # Determine ground truth field
    if eval_mode == "synthetic":
        gt_field = "reference"
    else:
        gt_field = "silver_reference"
        # Check if silver labels are available
        if not any(gt_field in item for item in benchmark):
            return None

    correct = 0
    total = 0
    match_flags = []

    for bench_item, res_item in zip(benchmark, results):
        ground_truth = bench_item.get(gt_field)
        extracted = get_extracted_text(res_item)

        gt_empty = ground_truth is None or normalize_text(ground_truth) == ""
        ext_empty = extracted is None or normalize_text(extracted) == ""

        if gt_empty and ext_empty:
            # Both empty — method correctly found no valid translation
            match_flags.append(True)
            correct += 1
            total += 1
        elif gt_empty or ext_empty:
            # One empty, one not — mismatch
            match_flags.append(False)
            total += 1
        else:
            matched = is_exact_match(extracted, ground_truth)
            match_flags.append(matched)
            correct += int(matched)
            total += 1

    metrics = {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }

    # Per-category breakdown
    breakdown = compute_extraction_breakdown(benchmark, match_flags)
    if breakdown:
        metrics["breakdown"] = breakdown

    # Average coverage for curated mode
    if eval_mode == "curated":
        coverages = []
        for bench_item, res_item in zip(benchmark, results):
            extracted = get_extracted_text(res_item)
            if extracted is not None:
                original = bench_item.get("translation", "")
                coverages.append(compute_coverage(extracted, original))
        if coverages:
            metrics["avg_coverage"] = sum(coverages) / len(coverages)

    return metrics


# ─────────────────────────────────────────────────────────────
# Noise prediction evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_noise_prediction(
    benchmark: List[Dict],
    results: List[Dict],
) -> Optional[Dict]:
    """
    Evaluate noise pattern/category prediction (LLM extractor only).

    Only runs if results contain 'predicted_noise_pattern' and benchmark
    has 'noise_pattern' or 'noise_category' gold labels.
    """
    # Check if results have predictions
    has_predictions = any(
        r.get("predicted_noise_pattern") is not None for r in results
    )
    if not has_predictions:
        return None

    # Check if benchmark has gold labels
    has_gold_pattern = any("noise_pattern" in item for item in benchmark)
    has_gold_category = any("noise_category" in item for item in benchmark)
    if not has_gold_pattern and not has_gold_category:
        return None

    # Only evaluate on noisy instances (label=0)
    pattern_correct = 0
    category_correct = 0
    total_noisy = 0

    # For confusion matrix
    category_confusion = defaultdict(lambda: defaultdict(int))

    for bench_item, res_item in zip(benchmark, results):
        if bench_item.get("label", 0) != 0:
            continue

        total_noisy += 1
        pred_pattern = res_item.get("predicted_noise_pattern")
        # Fallback: parse noise_pattern from extracted_text JSON if field is null
        if not pred_pattern and res_item.get("extracted_text"):
            parsed = _parse_llm_json(res_item["extracted_text"])
            if parsed:
                pred_pattern = parsed.get("noise_pattern")

        # Pattern accuracy
        if has_gold_pattern and pred_pattern:
            gold_pattern = bench_item.get("noise_pattern", "")
            if pred_pattern == gold_pattern:
                pattern_correct += 1

        # Category accuracy (derived from pattern)
        if has_gold_category and pred_pattern:
            pred_category = derive_noise_category(pred_pattern)
            gold_category = bench_item.get("noise_category", "")
            if pred_category == gold_category:
                category_correct += 1
            # Confusion matrix
            if gold_category and pred_category:
                category_confusion[gold_category][pred_category] += 1

    if total_noisy == 0:
        return None

    result = {"total_noisy": total_noisy}

    if has_gold_pattern:
        result["pattern_accuracy"] = pattern_correct / total_noisy
        result["pattern_correct"] = pattern_correct

    if has_gold_category:
        result["category_accuracy"] = category_correct / total_noisy
        result["category_correct"] = category_correct
        result["category_confusion"] = dict(category_confusion)

    return result


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

def print_results(
    method_detection: Dict[str, Dict],
    method_detection_breakdowns: Dict[str, Dict],
    method_extraction: Dict[str, Optional[Dict]],
    method_noise_preds: Dict[str, Dict],
    eval_mode: str,
):
    """Print evaluation results."""
    # ── Detection Accuracy ──
    print(f"\n{'=' * 80}")
    print(f"Detection Accuracy (extracted != original → noisy)")
    print(f"Eval mode: {eval_mode}")
    print(f"{'=' * 80}\n")

    header = (
        f"{'Method':<35} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'Accuracy':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    )
    print(header)
    print("-" * len(header))

    for method, metrics in method_detection.items():
        print(
            f"{method:<35} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['accuracy']:>10.4f} "
            f"{metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6} {metrics['tn']:>6}"
        )

    # Per-category detection breakdown
    for method, breakdown in method_detection_breakdowns.items():
        if not breakdown:
            continue
        print(f"\n  {method} -- detection breakdown by noise category:")
        for ptype, metrics in sorted(breakdown.items()):
            n = metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]
            print(
                f"    {ptype:<20} P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  (n={n})"
            )

    # ── Extraction Accuracy ──
    has_extraction = any(v is not None for v in method_extraction.values())
    if has_extraction:
        print(f"\n{'=' * 80}")
        print(f"Extraction Accuracy (extracted == ground truth)")
        gt_desc = "reference" if eval_mode == "synthetic" else "silver_reference"
        print(f"Ground truth: {gt_desc}")
        print(f"{'=' * 80}\n")

        header = f"{'Method':<35} {'Accuracy':>10} {'Correct':>10} {'Total':>10}"
        has_coverage = any(
            v is not None and "avg_coverage" in v
            for v in method_extraction.values()
        )
        if has_coverage:
            header += f" {'Avg Coverage':>14}"
        print(header)
        print("-" * len(header))

        for method, metrics in method_extraction.items():
            if metrics is None:
                continue
            line = (
                f"{method:<35} {metrics['accuracy']:>10.4f} "
                f"{metrics['correct']:>10} {metrics['total']:>10}"
            )
            if has_coverage and "avg_coverage" in metrics:
                line += f" {metrics['avg_coverage']:>14.4f}"
            print(line)

        # Per-category extraction breakdown
        for method, metrics in method_extraction.items():
            if metrics is None:
                continue
            breakdown = metrics.get("breakdown", {})
            if not breakdown:
                continue
            print(f"\n  {method} -- extraction breakdown by noise category:")
            for ptype, cat_metrics in sorted(breakdown.items()):
                print(
                    f"    {ptype:<20} accuracy={cat_metrics['accuracy']:.4f}  "
                    f"({cat_metrics['correct']}/{cat_metrics['total']})"
                )
    else:
        if eval_mode == "curated":
            print(f"\n  [Extraction accuracy skipped: no silver_reference in benchmark. "
                  f"Run aggregate_silver_labels.py first.]")

    # ── Noise prediction evaluation ──
    for method, noise_pred in method_noise_preds.items():
        if not noise_pred:
            continue

        total = noise_pred["total_noisy"]
        print(f"\n{'=' * 80}")
        print(f"Noise Type Prediction ({method})")
        print(f"{'=' * 80}")

        if "pattern_accuracy" in noise_pred:
            print(
                f"  Pattern accuracy:   {noise_pred['pattern_accuracy']:.4f}  "
                f"({noise_pred['pattern_correct']} / {total})"
            )
        if "category_accuracy" in noise_pred:
            print(
                f"  Category accuracy:  {noise_pred['category_accuracy']:.4f}  "
                f"({noise_pred['category_correct']} / {total})"
            )

        # Confusion matrix
        confusion = noise_pred.get("category_confusion", {})
        if confusion:
            all_cats = sorted(
                set(list(confusion.keys())
                    + [c for row in confusion.values() for c in row.keys()])
            )
            print(f"\n  Category confusion matrix:")
            print(f"  {'Gold \\ Pred':<15}", end="")
            for cat in all_cats:
                print(f"  {cat:>12}", end="")
            print()
            for gold_cat in all_cats:
                print(f"  {gold_cat:<15}", end="")
                for pred_cat in all_cats:
                    count = confusion.get(gold_cat, {}).get(pred_cat, 0)
                    print(f"  {count:>12}", end="")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clean translation detection results against a benchmark"
    )
    parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark JSONL with gold labels (dev.jsonl, test.jsonl, or curated_noisy_1100_with_silver.jsonl)",
    )
    parser.add_argument(
        "--results", nargs="+", type=str, required=True,
        help="One or more result JSONL files from individual method runs",
    )
    parser.add_argument(
        "--eval-mode", choices=["synthetic", "curated", "auto"], default="auto",
        help="Evaluation mode: synthetic (exact match), curated (silver labels), or auto-detect",
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON file for metrics",
    )
    args = parser.parse_args()

    if not Path(args.benchmark).exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    benchmark = load_jsonl(args.benchmark)
    gold_labels = [item.get("label", 0) for item in benchmark]
    print(f"Loaded benchmark: {len(benchmark)} instances "
          f"({sum(gold_labels)} clean, {len(gold_labels) - sum(gold_labels)} noisy)")

    # Determine eval mode
    if args.eval_mode == "auto":
        eval_mode = detect_eval_mode(benchmark)
        print(f"Auto-detected eval mode: {eval_mode}")
    else:
        eval_mode = args.eval_mode

    method_detection = {}
    method_detection_breakdowns = {}
    method_extraction = {}
    method_noise_preds = {}

    for results_path in args.results:
        if not Path(results_path).exists():
            print(f"Warning: Results file not found: {results_path}, skipping")
            continue

        method_name = infer_method_name(results_path)
        results = load_jsonl(results_path)

        if len(results) != len(benchmark):
            print(f"Warning: {method_name} has {len(results)} results but benchmark "
                  f"has {len(benchmark)} instances. Skipping.")
            continue

        print(f"\nEvaluating: {method_name} ({len(results)} results)")

        # Detection accuracy
        det_pred_labels, det_metrics = evaluate_detection(benchmark, results)
        method_detection[method_name] = det_metrics

        # Detection breakdown by category
        det_breakdown = compute_breakdown(benchmark, gold_labels, det_pred_labels)
        method_detection_breakdowns[method_name] = det_breakdown

        # Extraction accuracy
        ext_metrics = evaluate_extraction(benchmark, results, eval_mode)
        method_extraction[method_name] = ext_metrics

        # Noise prediction evaluation
        noise_pred = evaluate_noise_prediction(benchmark, results)
        method_noise_preds[method_name] = noise_pred

    if not method_detection:
        print("Error: No valid results to evaluate.")
        sys.exit(1)

    print_results(
        method_detection, method_detection_breakdowns,
        method_extraction, method_noise_preds, eval_mode,
    )

    # Save metrics to JSON
    if args.output:
        output_data = {
            "benchmark": args.benchmark,
            "eval_mode": eval_mode,
            "num_instances": len(benchmark),
            "methods": {},
        }
        for method in method_detection:
            entry = {
                "detection": method_detection[method],
            }
            if method in method_detection_breakdowns and method_detection_breakdowns[method]:
                entry["detection"]["breakdown"] = method_detection_breakdowns[method]
            if method_extraction.get(method) is not None:
                entry["extraction"] = method_extraction[method]
            if method_noise_preds.get(method) is not None:
                entry["noise_prediction"] = method_noise_preds[method]
            output_data["methods"][method] = entry

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
