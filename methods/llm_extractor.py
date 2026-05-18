#!/usr/bin/env python3
"""
Method: LLM-as-Extractor for clean translation detection.

Prompts a small instruction-following LLM (via vLLM) to extract only the
translation from each noisy LLM output. Classification is based on
coverage ratio between extracted text and original output.
"""

import json
import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict
from vllm import LLM, SamplingParams


EXTRACTION_PROMPT = (
    "Analyze this machine translation output. The source language is {src_lang} "
    "and the target language is {tgt_lang}.\n"
    "\n"
    "1. Extract ONLY the clean translation (no explanations, labels, or formatting).\n"
    "2. Identify the noise pattern if the output contains noise.\n"
    "\n"
    "Return a JSON object with these fields:\n"
    '- "extracted_translation": the clean translation text only\n'
    '- "noise_pattern": one of "none", "language_prefix", "verbose_preamble", '
    '"translation_prefix", "explanation", "cultural_note", "bilingual_output", '
    '"alternatives", "code_block", "special_formatting", "extra_punctuation", '
    '"wrong_language", "off_topic"\n'
    "\n"
    "Machine translation output:\n"
    "{translation}\n"
    "\n"
    "JSON:"
)


def build_messages(
    translations: List[str],
    src_langs: List[str],
    tgt_langs: List[str],
) -> List[List[Dict[str, str]]]:
    """Build chat-format messages for each instance."""
    messages_list = []
    for trans, sl, tl in zip(translations, src_langs, tgt_langs):
        user_content = EXTRACTION_PROMPT.format(
            src_lang=sl, tgt_lang=tl, translation=trans
        )
        messages_list.append([{"role": "user", "content": user_content}])
    return messages_list


def classify_batch(
    sources: List[str],
    translations: List[str],
    src_langs: List[str],
    tgt_langs: List[str],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    coverage_threshold: float = 0.8,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
) -> List[Dict]:
    """
    Classify a batch of translations using an LLM extractor.

    Args:
        sources: Source texts.
        translations: LLM translation outputs (potentially noisy).
        src_langs: Source language codes per instance.
        tgt_langs: Target language codes per instance.
        model_name: vLLM model identifier.
        coverage_threshold: Min coverage ratio to classify as clean. NOT USED.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum model context length.

    Returns:
        List of classification dicts matching classify_translation() interface,
        plus extra fields: extracted_text, coverage.
    """
    print(f"Initializing vLLM with model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    print("vLLM initialized")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        top_p=1.0,
    )

    messages_list = build_messages(translations, src_langs, tgt_langs)

    print(f"Running extraction on {len(messages_list)} instances...")
    outputs = llm.chat(
        messages=messages_list,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    print(f"Extraction complete for {len(outputs)} instances")

    results = []
    for trans, output in zip(translations, outputs):
        raw_output = output.outputs[0].text.strip()
        extracted = raw_output

        # Strip thinking/reasoning content from thinking models (e.g. Qwen3)
        # Pattern 1: XML-style <think>...</think> tags
        think_match = re.search(r"</think>\s*", extracted)
        if think_match:
            extracted = extracted[think_match.end():].strip()
        # Pattern 2: Plain-text "Thinking Process:" or similar headers
        think_text_match = re.match(
            r"(?:Thinking Process|思考过程|Thought Process|My Thinking)[:\s]",
            extracted,
            re.IGNORECASE,
        )
        if think_text_match:
            paragraphs = re.split(r"\n\s*\n", extracted)
            last_paragraph = ""
            for p in reversed(paragraphs):
                if p.strip():
                    last_paragraph = p.strip()
                    break
            extracted = last_paragraph

        # Try to parse as JSON
        predicted_noise_pattern = None
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                extracted_text = parsed.get("extracted_translation", "").strip()
                predicted_noise_pattern = parsed.get("noise_pattern")
                if extracted_text:
                    extracted = extracted_text
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat entire output as extracted text (current behavior)
            pass

        full_len = len(trans.strip())
        ext_len = len(extracted)
        coverage = ext_len / full_len if full_len > 0 else 1.0

        results.append({
            "extracted_text": extracted,
            "predicted_noise_pattern": predicted_noise_pattern,
            "coverage": coverage,
            "raw_llm_output": raw_output,
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
        description="LLM-as-extractor for clean translation detection"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-122B-A10B",
        help="LLM model name for extraction",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Coverage threshold for clean classification",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=262144,
        help="Maximum model context length",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} instances from {args.input}")

    sources = [item["source"] for item in data]
    translations = [item["translation"] for item in data]
    src_langs = [item["src_lang"] for item in data]
    tgt_langs = [item["tgt_lang"] for item in data]

    results = classify_batch(
        sources,
        translations,
        src_langs,
        tgt_langs,
        model_name=args.model,
        coverage_threshold=args.coverage_threshold,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item, res in zip(data, results):
            out = {**item, **res}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")
    print(f"  Instances with noise pattern prediction: "
          f"{sum(1 for r in results if r['predicted_noise_pattern'] is not None)}/{len(results)}")


if __name__ == "__main__":
    main()
