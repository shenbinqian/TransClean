"""
check_silver_agreement.py
=========================
Check agreement between silver label models and save inspection files.

Computes majority vote agreement statistics across 3 silver label files and
saves tiebreak and no-extraction instances as xlsx for human review.

Usage:
    python bench_creating/check_silver_agreement.py \
        --curated data/curated_noisy_1100.jsonl \
        --silver-files data/silver_labels/gpt5_mini.jsonl \
                       data/silver_labels/Qwen_Qwen3.5_122B_A10B.jsonl \
                       data/silver_labels/google_gemma_4_31B_it.jsonl \
        --proprietary-index 0 \
        --output-dir data/silver_label_agreement
"""

import json
import sys
import argparse
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from methods.evaluation_utils import normalize_text


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def majority_vote(
    extractions: list[str | None],
    proprietary_index: int = 0,
) -> tuple[str | None, int]:
    """Return (chosen_translation, agreement_count).

    agreement_count: 3 = all agree, 2 = two agree, 1 = none agree (tiebreak),
                     0 = no model produced an extraction.
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

    # No agreement — use proprietary model's result if non-empty, else first non-empty
    if normalized[proprietary_index] is not None:
        return extractions[proprietary_index], 1
    for i in range(n):
        if normalized[i] is not None:
            return extractions[i], 1
    return None, 0


def _save_inspection_xlsx(
    no_extraction_rows: list,
    tiebreak_rows: list,
    silver_models: list[str],
    output_dir: Path,
):
    """Save no-extraction and tiebreak instances as xlsx for human inspection."""
    headers = [
        "idx", "id", "lang_pair", "source", "translation",
        "reference", "primary_pattern",
        f"extraction_{silver_models[0]}" if len(silver_models) > 0 else "extraction_m0",
        f"extraction_{silver_models[1]}" if len(silver_models) > 1 else "extraction_m1",
        f"extraction_{silver_models[2]}" if len(silver_models) > 2 else "extraction_m2",
    ]

    def _write_wb(path: Path, rows, has_chosen: bool):
        wb = openpyxl.Workbook()
        ws = wb.active
        cols = headers + (["chosen"] if has_chosen else [])
        ws.append(cols)
        for row in rows:
            if has_chosen:
                idx, rec, exts, chosen = row
            else:
                idx, rec, exts = row
            values = [
                idx,
                rec.get("id", ""),
                rec.get("lang_pair", ""),
                rec.get("source", ""),
                rec.get("translation", ""),
                rec.get("reference", ""),
                rec.get("primary_pattern", ""),
                exts[0] if exts[0] else "",
                exts[1] if exts[1] else "",
                exts[2] if exts[2] else "",
            ]
            if has_chosen:
                values.append(chosen if chosen else "")
            ws.append(values)
        wb.save(path)
        print(f"  Saved {len(rows)} rows to {path}")

    if no_extraction_rows:
        _write_wb(output_dir / "no_extraction_instances.xlsx", no_extraction_rows, has_chosen=False)
    if tiebreak_rows:
        _write_wb(output_dir / "tiebreak_instances.xlsx", tiebreak_rows, has_chosen=True)


def main():
    parser = argparse.ArgumentParser(
        description="Check silver label agreement and save inspection files."
    )
    parser.add_argument("--curated", required=True,
                        help="Path to original curated_noisy_1100.jsonl")
    parser.add_argument("--silver-files", nargs=3, required=True,
                        help="3 silver label JSONL files (proprietary first)")
    parser.add_argument("--proprietary-index", type=int, default=0,
                        help="Index of the proprietary model in --silver-files (default: 0)")
    parser.add_argument("--output-dir", default="data/silver_label_agreement",
                        help="Directory for output xlsx and stats (default: data/silver_label_agreement)")
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

    # Check agreement
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agreement_counts = {3: 0, 2: 0, 1: 0, 0: 0}
    no_extraction_rows = []
    tiebreak_rows = []

    for idx in range(len(curated)):
        extractions = [
            silver_data[m][idx].get("silver_extracted_translation")
            for m in range(3)
        ]

        chosen, agreement = majority_vote(extractions, args.proprietary_index)
        agreement_counts[agreement] += 1

        if chosen is None:
            no_extraction_rows.append((idx, curated[idx], extractions))
        elif agreement == 1:
            tiebreak_rows.append((idx, curated[idx], extractions, chosen))

    # Print statistics
    total = len(curated)
    print(f"\n{'='*60}")
    print(f"Silver Label Agreement Check")
    print(f"{'='*60}")
    print(f"  Total instances:  {total}")
    print(f"  3/3 agreement:    {agreement_counts[3]} ({agreement_counts[3]/total*100:.1f}%)")
    print(f"  2/3 agreement:    {agreement_counts[2]} ({agreement_counts[2]/total*100:.1f}%)")
    print(f"  1/3 (tiebreak):   {agreement_counts[1]} ({agreement_counts[1]/total*100:.1f}%)")
    print(f"  No extraction:    {agreement_counts[0]} ({agreement_counts[0]/total*100:.1f}%)")
    print(f"{'='*60}")

    # Save xlsx files for human inspection
    _save_inspection_xlsx(no_extraction_rows, tiebreak_rows, silver_models, output_dir)


if __name__ == "__main__":
    main()
