"""
generate_silver_labels_vllm.py
==============================
Generate silver labels (extracted clean translations) for curated noisy data
using open-weight LLMs via vLLM.

Part of the 3-model silver label pipeline:
  1. generate_silver_labels_api.py (proprietary model)
  2. This script (open-weight models, e.g. Qwen3.5-122B, Gemma-4-31B)
  3. aggregate_silver_labels.py (majority vote)

Usage:
    python generate_silver_labels_vllm.py \\
        --input data/curated_noisy_1100.jsonl \\
        --output data/silver_labels/Qwen_Qwen3.5_122B_A10B.jsonl \\
        --model Qwen/Qwen3.5-122B-A10B \\
        --tensor-parallel-size 4

    python generate_silver_labels_vllm.py \\
        --input data/curated_noisy_1100.jsonl \\
        --output data/silver_labels/google_gemma_4_31B_it.jsonl \\
        --model google/gemma-4-31B-it \\
        --tensor-parallel-size 4
"""

import json
import argparse
from pathlib import Path

from silver_label_prompt import (
    SILVER_LABEL_SYSTEM_PROMPT,
    build_silver_label_prompt,
    parse_silver_label_response,
)


def build_chat_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": SILVER_LABEL_SYSTEM_PROMPT},
        {"role": "user",   "content": build_silver_label_prompt(record)},
    ]


def run_vllm_batch(
    records: list[dict],
    model_name: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    temperature: float,
    batch_size: int,
) -> list[str]:
    """Run all records through vLLM and return raw response strings."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Build all chat prompts using the model's chat template
    tokenizer = llm.get_tokenizer()
    # For Qwen3-class models: disable thinking via chat_template_kwargs
    apply_kwargs: dict = {}
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            apply_kwargs["enable_thinking"] = False
        except TypeError:
            pass

    prompts: list[str] = []
    for record in records:
        messages = build_chat_messages(record)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        prompts.append(prompt)

    print(f"Running vLLM inference on {len(prompts)} prompts "
          f"(batch_size={batch_size}, temp={temperature}, max_tokens={max_tokens})")

    raw_outputs: list[str] = [""] * len(prompts)
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch = prompts[start:end]
        outputs = llm.generate(batch, sampling_params)
        for i, out in enumerate(outputs):
            raw_outputs[start + i] = out.outputs[0].text
        print(f"  Processed {end}/{len(prompts)}", flush=True)

    return raw_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate silver labels using an open-weights model via vLLM."
    )
    parser.add_argument("--input", required=True,
                        help="Path to curated_noisy_1100.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B",
                        help="HuggingFace model ID to run via vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max context length for vLLM (default: 8192)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max generation tokens per response (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of prompts per vLLM generate call (default: 256)")
    args = parser.parse_args()

    # Load curated examples
    records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    # Run vLLM
    raw_outputs = run_vllm_batch(
        records=records,
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    # Parse and write results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_success = 0
    n_failed = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for record, raw in zip(records, raw_outputs):
            extracted = parse_silver_label_response(raw)

            out_record = dict(record)
            out_record["silver_model"] = args.model
            out_record["silver_extracted_translation"] = extracted
            out_record["silver_raw_response"] = raw
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if extracted is not None:
                n_success += 1
            else:
                n_failed += 1

    print(f"\n{'='*60}")
    print(f"DONE: {len(records)} records processed")
    print(f"  Extracted successfully: {n_success}")
    print(f"  Parse failures:        {n_failed}")
    print(f"  Saved to:              {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
