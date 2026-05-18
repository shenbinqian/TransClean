"""
build_benchmark.py
==================
Orchestrator script that builds the full translation noise benchmark.

Steps:
  1. Sample clean data from raw_data/ or use existing clean_samples.jsonl
  2. Generate noisy variants (3 per clean sample) using generate_noisy_translations.py or use existing noisy_variants.jsonl
  3. Merge clean + noisy, assign IDs and labels
  4. Stratified dev/test split (50/50)
  5. Write data/dev.jsonl and data/test.jsonl
  6. Print statistics

Usage:
    python bench_creating/build_benchmark.py --api-key YOUR_KEY
    python bench_creating/build_benchmark.py --api-key YOUR_KEY --samples-per-lp 5  # small test
"""

import random
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter

# Import from sibling modules
from sample_clean_data import sample_clean_data
from generate_noisy_translations import (
    generate_noisy_for_category,
    generate_combo_noise,
    save_jsonl,
    load_jsonl,
    create_llm_client,
    DEFAULT_MODEL,
)


def generate_all_noisy_variants(
    clean_samples: list[dict],
    client,
    delay: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Generate 3 noisy variants per clean sample."""
    random.seed(seed)
    all_noisy = []
    total = len(clean_samples)

    for i, record in enumerate(clean_samples):
        if i % 50 == 0:
            print(f"  [{i+1}/{total}] Generating 3 noisy variants...")

        # 1. formatting
        result = generate_noisy_for_category(record, "formatting", client)
        all_noisy.append({
            **record,
            "noisy_translation": result["noisy_translation"],
            "noise_category": "formatting",
            "noise_pattern": result.get("noise_pattern", "unknown"),
        })

        # 2. content (LLM)
        result = generate_noisy_for_category(record, "content", client)
        all_noisy.append({
            **record,
            "noisy_translation": result["noisy_translation"],
            "noise_category": "content",
            "noise_pattern": result.get("noise_pattern", "unknown"),
        })
        if delay > 0:
            time.sleep(delay)

        # 3. combo
        noisy_text, combo_label, pattern_label = generate_combo_noise(record, client)
        all_noisy.append({
            **record,
            "noisy_translation": noisy_text,
            "noise_category": "combo",
            "noise_pattern": pattern_label,
            "combo_categories": combo_label,
        })
        if delay > 0:
            time.sleep(delay)

    return all_noisy


def stratified_split(
    clean_samples: list[dict],
    noisy_samples: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split into dev/test (50/50), stratified by LP.
    Each clean sample's 3 noisy variants follow the same split.
    """
    random.seed(seed)

    # Precompute LP-local index for each global index
    lp_counter = Counter()
    idx_to_lp_idx = {}
    for i, rec in enumerate(clean_samples):
        lp = rec["lp"]
        idx_to_lp_idx[i] = lp_counter[lp]
        lp_counter[lp] += 1

    # Group clean sample indices by LP
    lp_groups = defaultdict(list)
    for i, rec in enumerate(clean_samples):
        lp_groups[rec["lp"]].append(i)

    # Shuffle within each LP and split 50/50
    dev_indices = set()
    for lp, indices in sorted(lp_groups.items()):
        shuffled = list(indices)
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        dev_indices.update(shuffled[:mid])

    dev_records = []
    test_records = []

    def make_entry(rec, lp_idx, category, is_clean, noise_pattern="none", noisy_text=None):
        label_suffix = "clean" if is_clean else category
        return {
            "id": f"{rec['lp']}_{lp_idx:03d}_{label_suffix}",
            "lp": rec["lp"],
            "source": rec["source"],
            "reference": rec["reference"],
            "translation": rec["reference"] if is_clean else noisy_text,
            "src_lang": rec["src_lang"],
            "tgt_lang": rec["tgt_lang"],
            "label": 1 if is_clean else 0,
            "noise_category": "clean" if is_clean else category,
            "noise_pattern": noise_pattern,
        }

    # Add clean samples
    for i, rec in enumerate(clean_samples):
        lp_idx = idx_to_lp_idx[i]
        entry = make_entry(rec, lp_idx, "clean", is_clean=True)
        if i in dev_indices:
            entry["split"] = "dev"
            dev_records.append(entry)
        else:
            entry["split"] = "test"
            test_records.append(entry)

    # Add noisy samples (3 per clean sample, follow same split)
    for i, rec in enumerate(noisy_samples):
        clean_idx = i // 3
        lp_idx = idx_to_lp_idx[clean_idx]
        category = rec["noise_category"]

        entry = make_entry(
            rec, lp_idx, category, is_clean=False,
            noise_pattern=rec.get("noise_pattern", "unknown"),
            noisy_text=rec["noisy_translation"],
        )

        if clean_idx in dev_indices:
            entry["split"] = "dev"
            dev_records.append(entry)
        else:
            entry["split"] = "test"
            test_records.append(entry)

    return dev_records, test_records


def print_statistics(dev: list[dict], test: list[dict]):
    """Print benchmark statistics."""
    print(f"\n{'=' * 60}")
    print("Benchmark Statistics")
    print(f"{'=' * 60}")
    print(f"  Dev:  {len(dev)} instances")
    print(f"  Test: {len(test)} instances")
    print(f"  Total: {len(dev) + len(test)} instances")

    for split_name, data in [("Dev", dev), ("Test", test)]:
        clean = sum(1 for r in data if r["label"] == 1)
        noisy = sum(1 for r in data if r["label"] == 0)
        print(f"\n  {split_name}:")
        print(f"    Clean: {clean}")
        print(f"    Noisy: {noisy}")

        cats = Counter(r["noise_category"] for r in data if r["label"] == 0)
        for cat, count in sorted(cats.items()):
            print(f"      {cat}: {count}")

        lps = Counter(r["lp"] for r in data)
        print(f"    Language pairs: {len(lps)}")


def main():
    parser = argparse.ArgumentParser(description="Build the translation noise benchmark")
    parser.add_argument("--raw-data-dir", default="raw_data", help="Path to raw_data directory")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--samples-per-lp", type=int, default=100, help="Samples per language pair")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default=None, help="Model name (default: gpt-5-mini-2025-08-07 for openai)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Skip sampling step (use existing clean_samples.jsonl)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation step (use existing noisy_variants.jsonl)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Sample clean data
    clean_path = out_dir / "clean_samples.jsonl"
    if args.skip_sampling and clean_path.exists():
        print("Step 1: Loading existing clean samples...")
        clean_samples = load_jsonl(str(clean_path))
    else:
        print("Step 1: Sampling clean data...")
        clean_samples = sample_clean_data(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            samples_per_lp=args.samples_per_lp,
            seed=args.seed,
        )

    print(f"  → {len(clean_samples)} clean samples")

    # Step 2: Generate noisy variants
    noisy_path = out_dir / "noisy_variants.jsonl"
    if args.skip_generation and noisy_path.exists():
        print("\nStep 2: Loading existing noisy variants...")
        noisy_samples = load_jsonl(str(noisy_path))
    else:
        print("\nStep 2: Generating noisy variants...")
        model = args.model or (DEFAULT_MODEL if args.provider == "openai" else "claude-3-5-haiku-20241022")
        client = create_llm_client(
            provider=args.provider, api_key=args.api_key, model=model,
        )
        print(f"  {args.provider.capitalize()} client initialized. Model: {model}")

        noisy_samples = generate_all_noisy_variants(
            clean_samples, client, args.delay, args.seed,
        )
        save_jsonl(noisy_samples, str(noisy_path))

    print(f"  → {len(noisy_samples)} noisy variants")

    # Step 3 & 4: Merge, assign IDs, and split
    print("\nStep 3: Stratified dev/test split...")
    dev_records, test_records = stratified_split(
        clean_samples, noisy_samples, seed=args.seed,
    )

    # Step 5: Write output
    dev_path = out_dir / "dev.jsonl"
    test_path = out_dir / "test.jsonl"
    save_jsonl(dev_records, str(dev_path))
    save_jsonl(test_records, str(test_path))
    print(f"  → {dev_path}")
    print(f"  → {test_path}")

    # Step 6: Statistics
    print_statistics(dev_records, test_records)


if __name__ == "__main__":
    main()
