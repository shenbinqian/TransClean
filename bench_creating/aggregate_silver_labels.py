"""
aggregate_silver_labels.py
==========================
Aggregate silver labels from multiple LLMs via majority vote.

Takes 3 silver label JSONL files and produces a single output with the
agreed-upon clean translation as `silver_reference`.

Aggregation logic:
  - If 2+ models agree (after normalization), use the majority answer
  - If no agreement, use the trusted model's answer (including empty/None)

Usage:
    python bench_creating/aggregate_silver_labels.py \
        --curated data/curated_noisy_1100.jsonl \
        --silver-files data/silver_labels/gpt5_mini.jsonl \
                       data/silver_labels/Qwen_Qwen3.5_122B_A10B.jsonl \
                       data/silver_labels/google_gemma_4_31B_it.jsonl \
        --trusted-index 2 \
        --output data/curated_noisy_1100_with_silver.jsonl
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from methods.evaluation_utils import normalize_text


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def aggregate(
    extractions: list[str | None],
    trusted_index: int,
) -> tuple[str | None, int]:
    """Return (chosen_translation, agreement_count).

    agreement_count: 3 = all agree, 2 = two agree, 1 = no agreement (uses trusted model).
    When no agreement, the trusted model's result is used as-is, even if empty/None.
    """
    # Normalize all extractions for comparison; treat empty strings as None
    normalized = []
    for ext in extractions:
        if ext is not None and ext.strip() != "":
            normalized.append(normalize_text(ext))
        else:
            normalized.append(None)

    n = len(normalized)

    # Check for 3-way agreement
    if n >= 3 and normalized[0] == normalized[1] == normalized[2] and normalized[0] is not None:
        return extractions[0], 3

    # Check for 2-way agreement
    for i in range(n):
        for j in range(i + 1, n):
            if normalized[i] is not None and normalized[i] == normalized[j]:
                return extractions[i], 2

    # No agreement — use trusted model's result as-is (even if None/empty)
    trusted_ext = extractions[trusted_index]
    if trusted_ext is not None and trusted_ext.strip() == "":
        trusted_ext = None
    return trusted_ext, 1


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate silver labels from multiple LLMs via majority vote."
    )
    parser.add_argument("--curated", required=True,
                        help="Path to original curated_noisy_1100.jsonl")
    parser.add_argument("--silver-files", nargs=3, required=True,
                        help="3 silver label JSONL files")
    parser.add_argument("--trusted-index", type=int, default=2,
                        help="Index of the trusted model in --silver-files (default: 2, i.e. gemma4)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path with silver_reference field")
    args = parser.parse_args()

    # Load original curated data
    curated = load_jsonl(args.curated)
    print(f"Loaded {len(curated)} curated records from {args.curated}")

    # Load silver label files
    silver_data = []
    silver_models = []
    for path in args.silver_files:
        data = load_jsonl(path)
        silver_data.append(data)
        model_name = data[0].get("silver_model", Path(path).stem) if data else Path(path).stem
        silver_models.append(model_name)
        print(f"Loaded {len(data)} silver labels from {path} (model: {model_name})")

    # Validate lengths
    for i, data in enumerate(silver_data):
        if len(data) != len(curated):
            print(f"Error: {args.silver_files[i]} has {len(data)} records "
                  f"but curated has {len(curated)}. Aborting.")
            sys.exit(1)

    print(f"\nTrusted model (for tiebreaks): {silver_models[args.trusted_index]}")

    # Aggregate
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agreement_counts = {3: 0, 2: 0, 1: 0}

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx in range(len(curated)):
            extractions = [
                silver_data[m][idx].get("silver_extracted_translation")
                for m in range(3)
            ]
            votes = [ext if ext is not None else "" for ext in extractions]

            chosen, agreement = aggregate(extractions, args.trusted_index)
            agreement_counts[agreement] += 1

            out_record = dict(curated[idx])
            out_record["silver_reference"] = chosen
            out_record["silver_agreement"] = agreement
            out_record["silver_votes"] = votes
            out_record["silver_models"] = silver_models
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    # Print statistics
    total = len(curated)
    print(f"\n{'='*60}")
    print(f"Silver Label Aggregation Complete")
    print(f"{'='*60}")
    print(f"  Total instances:  {total}")
    print(f"  3/3 agreement:    {agreement_counts[3]} ({agreement_counts[3]/total*100:.1f}%)")
    print(f"  2/3 agreement:    {agreement_counts[2]} ({agreement_counts[2]/total*100:.1f}%)")
    print(f"  No agreement:     {agreement_counts[1]} ({agreement_counts[1]/total*100:.1f}%) → used {silver_models[args.trusted_index]}")
    print(f"  Saved to:         {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
