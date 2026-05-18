"""
generate_silver_labels_api.py
=============================
Generate silver labels (extracted clean translations) for curated noisy data
using a proprietary LLM via API (OpenAI or Anthropic).

Part of the 3-model silver label pipeline:
  1. This script (proprietary model, e.g. gpt-5-mini)
  2. generate_silver_labels_vllm.py (open-weight models via vLLM)
  3. aggregate_silver_labels.py (majority vote)

Usage:
    python bench_creating/generate_silver_labels_api.py \\
        --input data/curated_noisy_1100.jsonl \\
        --output data/silver_labels/gpt5_mini.jsonl \\
        --api-key YOUR_OPENAI_KEY
"""

import json
import os
import argparse
import time
from pathlib import Path

from curate_noisy_examples import LLMClient
from silver_label_prompt import (
    SILVER_LABEL_SYSTEM_PROMPT,
    build_silver_label_prompt,
    parse_silver_label_response,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate silver labels for curated noisy translations using a proprietary LLM."
    )
    parser.add_argument("--input", required=True,
                        help="Path to curated_noisy_1100.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--api-key", default=None,
                        help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    parser.add_argument("--model", default=None,
                        help="Model to use (default: gpt-5-mini-2025-08-07 for OpenAI, "
                             "claude-3-5-haiku-20241022 for Anthropic)")
    parser.add_argument("--max-tokens", type=int, default=9216,
                        help="Max tokens for LLM response (default: 9216)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds between API calls (rate limiting)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per API call")
    args = parser.parse_args()

    # Resolve model default
    if args.model is None:
        args.model = ("gpt-5-mini-2025-08-07" if args.provider == "openai"
                      else "claude-3-5-haiku-20241022")

    # Set up client
    api_key = args.api_key or os.environ.get(
        "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    )
    if not api_key:
        raise ValueError("API key required. Set env var or pass --api-key.")
    client = LLMClient(provider=args.provider, api_key=api_key, model=args.model)
    print(f"LLM client: {args.provider} / {args.model}")

    # Load curated data
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    # Process each record
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_success = 0
    n_failed = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            user_msg = build_silver_label_prompt(record)
            extracted = None
            raw_response = ""

            for attempt in range(args.max_retries):
                try:
                    raw_response = client.create(
                        system=SILVER_LABEL_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_msg}],
                        max_tokens=args.max_tokens,
                    )
                    extracted = parse_silver_label_response(raw_response)

                    if extracted is not None:
                        break
                    # Parse failed — retry
                    if attempt < args.max_retries - 1:
                        time.sleep(1)

                except Exception as e:
                    if attempt < args.max_retries - 1:
                        wait = 2 ** attempt
                        print(f"  [{idx}] retry {attempt+1}: {e}. Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  [{idx}] FAILED: {e}")

            out_record = dict(record)
            out_record["silver_model"] = args.model
            out_record["silver_extracted_translation"] = extracted
            out_record["silver_raw_response"] = raw_response
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if extracted is not None:
                n_success += 1
            else:
                n_failed += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(records)} "
                      f"(success={n_success}, failed={n_failed})")

            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"DONE: {len(records)} records processed")
    print(f"  Extracted successfully: {n_success}")
    print(f"  Parse failures:        {n_failed}")
    print(f"  Saved to:              {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
