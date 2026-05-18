"""
generate_noisy_translations.py
================================
Generates synthetic "noisy" LLM translation outputs from clean reference translations.

Based on patterns identified from 1M+ real LLM translation records across 17 models:
  1. Language-label prefix         e.g. "Chinese: 你好"
  2. Verbose preamble              e.g. "Here is the translation from English to Chinese: ..."
  3. Translation: / Translated:    e.g. "Translation: 你好"
  4. Newline explanation           e.g. translation + word breakdown / pinyin / grammar notes
  5. Cultural / contextual note    e.g. translation + "(Note: This idiom means...)"
  6. Bilingual output              e.g. "English: Hello\nChinese: 你好"
  7. Alternative translations      e.g. "1. option A\n2. option B\n3. option C"
  8. Markdown code block           e.g. "```\n你好\n```"
  9. Special formatting            e.g. "[[你好]]" or "{{你好}}"
 10. Extra punctuation             e.g. "你好!!!" (copied/amplified)
 11. Wrong language                e.g. translates into source language or a sibling language
 12. Off-topic                     e.g. AI boilerplate, code snippets, unrelated Q&A

Strategy:
  - Patterns 1, 3, 8, 9, 10           → rule-based (template, no LLM call needed)
  - Patterns 2, 4, 5, 6, 7, 11, 12   → LLM-based (content-aware, realistic)

Usage:
    # Generate with default settings (OpenAI, mixed rule-based + LLM)
    python generate_noisy_translations.py \\
        --input  path/to/test_translations.jsonl \\
        --output path/to/noisy_translations.jsonl \\
        --api-key YOUR_OPENAI_KEY \\
        --n      1000

    # Only rule-based (no LLM, free and fast)
    python generate_noisy_translations.py \\
        --input  path/to/test_translations.jsonl \\
        --output path/to/noisy_translations.jsonl \\
        --rule-based-only
"""

from __future__ import annotations

import json
import random
import os
import csv
import argparse
import time
from pathlib import Path

# ─────────────────────────────────────────────
# Unified LLM Client
# ─────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5-mini-2025-08-07"


class LLMClient:
    """Unified wrapper for Anthropic and OpenAI APIs."""

    def __init__(self, provider: str, api_key: str, model: str = DEFAULT_MODEL):
        self.provider = provider
        self.model = model
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    def create(self, system: str, messages: list[dict], max_tokens: int = 4096) -> str:
        """Send a chat completion request and return the text response."""
        if self.provider == "openai":
            oai_messages = [{"role": "system", "content": system}] + messages
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_tokens,
                messages=oai_messages,
            )
            return response.choices[0].message.content.strip()
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return response.content[0].text.strip()


def create_llm_client(
    provider: str = "openai",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> LLMClient:
    """Factory to create an LLM client with env var fallback for API key."""
    if provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass --api-key.")
    elif provider == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass --api-key.")
    return LLMClient(provider=provider, api_key=api_key, model=model)


from prompts import (
    LANG_NAMES,
    SYSTEM_PROMPT,
    LLM_PROMPT_TEMPLATES,
    build_llm_messages,
)

# Weights based on curated_noisy_1100.jsonl primary_pattern distribution.
# Rare patterns get floor weights to maintain diversity.
DEFAULT_PATTERN_WEIGHTS = {
    "explanation":         0.33,   # Pattern 4  (LLM)  curated: 35.7%
    "alternatives":        0.15,   # Pattern 7  (LLM)  curated: 17.5%
    "off_topic":           0.12,   # Pattern 12 (LLM)  curated: 12.9%  *** NEW ***
    "verbose_preamble":    0.09,   # Pattern 2  (LLM)  curated: 10.1%
    "bilingual_output":    0.07,   # Pattern 6  (LLM)  curated:  8.3%
    "wrong_language":      0.07,   # Pattern 11 (LLM)  curated:  7.2%  *** NEW ***
    "language_prefix":     0.07,   # Pattern 1  (rule) curated:  7.3%
    "extra_punctuation":   0.03,   # Pattern 10 (rule) floor
    "code_block":          0.03,   # Pattern 8  (rule) floor
    "special_formatting":  0.02,   # Pattern 9  (rule) floor
    "translation_prefix":  0.01,   # Pattern 3  (rule) floor
    "cultural_note":       0.01,   # Pattern 5  (LLM)  floor
}

LLM_PATTERNS = {"verbose_preamble", "explanation", "cultural_note", "bilingual_output", "alternatives",
                 "wrong_language", "off_topic"}
RULE_PATTERNS = set(DEFAULT_PATTERN_WEIGHTS.keys()) - LLM_PATTERNS


# ═══════════════════════════════════════════════════════════════════
# PART 1 — RULE-BASED GENERATORS
# ═══════════════════════════════════════════════════════════════════

def rule_language_prefix(translation: str, src_lang: str, tgt_lang: str, source: str) -> str:
    """
    Pattern 1: Prepend a language-name label before the translation.

    Examples (real outputs):
      "Chinese: 然而，28岁的特里斯坦并不打算不战而退"
      "Arabic: النص المترجم هنا"
    """
    lang_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())
    return f"{lang_name}: {translation}"


def rule_verbose_preamble(translation: str, src_lang: str, tgt_lang: str, source: str) -> str:
    """
    Pattern 2 (rule-based fallback): Introductory sentence before the translation.

    Examples (real outputs):
      "Here is the translation of '...' from English to Chinese:\n\n[translation]"
      "Translation from German to French:\n\n[translation]"

    Note: For more realistic output, use the LLM version (verbose_preamble).
    """
    src_name = LANG_NAMES.get(src_lang, src_lang.upper())
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())
    templates = [
        f'Here is the translation of "{source}" from {src_name} to {tgt_name}:\n\n{translation}',
        f'Here is the {tgt_name} translation:\n\n{translation}',
        f'Translation from {src_name} to {tgt_name}:\n\n{translation}',
        f'The {tgt_name} translation of the above is:\n\n{translation}',
    ]
    return random.choice(templates)


def rule_translation_prefix(translation: str, src_lang: str, tgt_lang: str, source: str) -> str:
    """
    Pattern 3: Simple "Translation:" or "Translated:" label prefix.

    Examples (real outputs):
      "Translation: 八项维和任务"
      "Translated: Das ist ein Beispiel."
    """
    prefix = random.choice(["Translation:", "Translated:", "Translation: ", "Here is the translation: "])
    return f"{prefix} {translation}"


def rule_code_block(translation: str, src_lang: str, tgt_lang: str, source: str) -> str:
    """
    Pattern 8: Wrap translation in markdown code block.

    Examples (real outputs):
      ```
      你好
      ```
    """
    return f"```\n{translation}\n```"


def rule_special_formatting(translation: str, src_lang: str, tgt_lang: str, source: str) -> str:
    """
    Pattern 9: Double brackets or braces around the translation.

    Examples (real outputs):
      "[[translation text]]"
      "{{translation text}}"
    """
    fmt = random.choice(["[[{}]]", "{{{}}}"])
    return fmt.format(translation)


def rule_extra_punctuation(translation: str, src_lang: str, tgt_lang: str, source: str) -> str:
    """
    Pattern 10: Repeated punctuation marks.

    Examples (real outputs):
      "我爱你!!!!"
      "你什么时候会关注孩子们???"
    """
    if translation.endswith("!"):
        return translation + random.choice(["!", "!!", "!!!"])
    elif translation.endswith("?"):
        return translation + random.choice(["?", "??"])
    else:
        # Force-add for variety on other sentences
        return translation + random.choice(["!!", "??"])


RULE_FUNCTIONS = {
    "language_prefix":    rule_language_prefix,
    "verbose_preamble":   rule_verbose_preamble,   # fallback — LLM version is better
    "translation_prefix": rule_translation_prefix,
    "code_block":         rule_code_block,
    "special_formatting": rule_special_formatting,
    "extra_punctuation":  rule_extra_punctuation,
}


# ═══════════════════════════════════════════════════════════════════
# CATEGORY ABSTRACTION
# ═══════════════════════════════════════════════════════════════════

CATEGORY_PATTERNS = {
    "formatting": {
        "patterns": ["language_prefix", "verbose_preamble", "translation_prefix",
                      "code_block", "special_formatting", "extra_punctuation"],
        "weights":  [0.07, 0.09, 0.01, 0.03, 0.02, 0.03],
    },
    "content": {
        "patterns": ["explanation", "cultural_note", "bilingual_output",
                      "alternatives", "wrong_language", "off_topic"],
        "weights":  [0.33, 0.01, 0.07, 0.15, 0.07, 0.12],
    },
}

# Natural wrappers for combo mode (subset of formatting patterns)
COMBO_FORMATTING_PATTERNS = ["language_prefix", "verbose_preamble", "translation_prefix"]
COMBO_FORMATTING_WEIGHTS = [0.28, 0.09, 0.03]


def pick_pattern_from_category(category: str) -> str:
    """Sample a sub-pattern from a category according to weights."""
    cat = CATEGORY_PATTERNS[category]
    return random.choices(cat["patterns"], weights=cat["weights"], k=1)[0]


def generate_combo_noise(
    record: dict,
    client: LLMClient | None = None,
) -> tuple[str, str, str]:
    """
    Generate combo noise by combining content + formatting.

    Returns (noisy_translation, category_label, pattern_label).
    Content pattern is applied first (content-modifying), then formatting wrapper.
    """
    combo_label = "content+formatting"

    # Step 1: Apply content category first
    expl_pattern = pick_pattern_from_category("content")
    noisy_text = generate_noisy(
        record, expl_pattern, client=client
    )["noisy_translation"]

    # Step 2: Apply formatting wrapper on top
    fmt_pattern = random.choices(
        COMBO_FORMATTING_PATTERNS, weights=COMBO_FORMATTING_WEIGHTS, k=1
    )[0]
    fmt_func = RULE_FUNCTIONS[fmt_pattern]
    src = record.get("src_lang", "en")
    tgt = record.get("tgt_lang", "zh")
    noisy_text = fmt_func(noisy_text, src, tgt, record["source"])
    pattern_label = f"{expl_pattern}+{fmt_pattern}"

    return noisy_text, combo_label, pattern_label


# ═══════════════════════════════════════════════════════════════════
# PART 2 — ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def pick_pattern(allowed_patterns: list[str]) -> str:
    """Sample a noise pattern according to its weight."""
    weights = [DEFAULT_PATTERN_WEIGHTS.get(p, 0.05) for p in allowed_patterns]
    total = sum(weights)
    normalized = [w / total for w in weights]
    return random.choices(allowed_patterns, weights=normalized, k=1)[0]


def generate_noisy(
    record: dict,
    pattern: str,
    client: LLMClient | None = None,
    max_retries: int = 3,
) -> dict:
    """
    Apply one noise pattern to a single translation record.

    Args:
        record:   dict with keys: source, reference, translation, src_lang, tgt_lang
        pattern:  one of the keys in DEFAULT_PATTERN_WEIGHTS
        client:   LLMClient instance (required for LLM patterns)
        max_retries: number of API retry attempts on failure

    Returns:
        A new dict with an added 'noisy_translation' and 'noise_pattern' field.
    """
    source = record["source"]
    clean  = record.get("reference", record.get("translation", ""))
    src    = record.get("src_lang", "en")
    tgt    = record.get("tgt_lang", "zh")

    result = dict(record)
    result["noise_pattern"] = pattern

    # ── Rule-based ──────────────────────────────────────────────────
    if pattern in RULE_FUNCTIONS:
        result["noisy_translation"] = RULE_FUNCTIONS[pattern](clean, src, tgt, source)
        return result

    # ── LLM-based ───────────────────────────────────────────────────
    if client is None:
        raise ValueError(
            f"Pattern '{pattern}' requires an LLM. Pass a client= argument or use --rule-based-only."
        )

    messages = build_llm_messages(pattern, source, src, tgt, clean)
    template = LLM_PROMPT_TEMPLATES.get(pattern, {})
    system = template.get("system_prompt_override", SYSTEM_PROMPT)

    for attempt in range(max_retries):
        try:
            result["noisy_translation"] = client.create(
                system=system,
                messages=messages,
                max_tokens=4096,
            )
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt+1}/{max_retries}] Error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {e}. Falling back to rule-based verbose_preamble.")
                result["noisy_translation"] = rule_verbose_preamble(clean, src, tgt, source)
                result["noise_pattern"] = "verbose_preamble_fallback"
                return result


def generate_noisy_for_category(
    record: dict,
    category: str,
    client: LLMClient | None = None,
) -> dict:
    """
    Generate a noisy variant for a specific category.

    Returns a dict with 'noisy_translation', 'noise_category', and 'noise_pattern'.
    """
    sub_pattern = pick_pattern_from_category(category)

    result = generate_noisy(record, sub_pattern, client=client)
    result["noise_category"] = category
    return result


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tsv(path: str) -> list[dict]:
    """Load records from a TSV file."""
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Unescape newlines
            for key in ("source", "reference"):
                if key in row and row[key]:
                    row[key] = row[key].replace("\\n", "\n")
            records.append(row)
    return records


def load_input(path: str) -> list[dict]:
    """Load records from JSONL or TSV based on file extension."""
    if path.endswith(".tsv"):
        return load_tsv(path)
    return load_jsonl(path)


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════
# PART 4 — CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy translations from clean reference translations."
    )
    parser.add_argument("--input",  required=True, help="Input JSONL file (with 'reference' or 'translation' field)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--n",      type=int, default=None, help="Max number of records to process (default: all)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model",   default=None, help="Model name (default: gpt-5-mini-2025-08-07 for openai, claude-3-5-haiku-20241022 for anthropic)")
    parser.add_argument("--rule-based-only", action="store_true",
                        help="Only apply rule-based patterns (no API calls)")
    parser.add_argument("--patterns", nargs="+", default=None,
                        choices=list(DEFAULT_PATTERN_WEIGHTS.keys()),
                        help="Restrict to specific noise patterns (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Seconds to wait between API calls (rate limiting)")
    parser.add_argument("--benchmark-mode", action="store_true",
                        help="Generate exactly 3 variants per record (formatting, content, combo)")
    args = parser.parse_args()

    random.seed(args.seed)

    client = None

    # Resolve default model based on provider
    if args.model is None:
        args.model = DEFAULT_MODEL if args.provider == "openai" else "claude-3-5-haiku-20241022"

    if args.benchmark_mode:
        # Benchmark mode always needs LLM
        client = create_llm_client(
            provider=args.provider, api_key=args.api_key, model=args.model,
        )
        print(f"{args.provider.capitalize()} client initialized (benchmark mode). Model: {args.model}")

        # Load data (supports TSV and JSONL)
        records = load_input(args.input)
        if args.n:
            records = records[:args.n]
        print(f"Loaded {len(records)} records from {args.input}")

        # Benchmark mode: generate 3 variants per record
        output_records = []
        category_counts = {"formatting": 0, "content": 0, "combo": 0}

        for i, record in enumerate(records):
            if i % 50 == 0:
                print(f"  [{i+1}/{len(records)}] generating 3 variants...")

            # 1. formatting
            result = generate_noisy_for_category(record, "formatting", client)
            result["noise_category"] = "formatting"
            output_records.append(result)
            category_counts["formatting"] += 1

            # 2. content
            result = generate_noisy_for_category(record, "content", client)
            result["noise_category"] = "content"
            output_records.append(result)
            category_counts["content"] += 1
            if args.delay > 0:
                time.sleep(args.delay)

            # 3. combo
            noisy_text, combo_label, pattern_label = generate_combo_noise(
                record, client
            )
            combo_result = dict(record)
            combo_result["noisy_translation"] = noisy_text
            combo_result["noise_category"] = "combo"
            combo_result["noise_pattern"] = pattern_label
            combo_result["combo_categories"] = combo_label
            output_records.append(combo_result)
            category_counts["combo"] += 1
            if args.delay > 0:
                time.sleep(args.delay)

        save_jsonl(output_records, args.output)
        print(f"\n✓ Saved {len(output_records)} noisy records to {args.output}")
        print("\nCategory distribution:")
        for cat, count in category_counts.items():
            print(f"  {cat:<25} {count:>5}")

    else:
        # Original mode
        allowed = args.patterns or list(DEFAULT_PATTERN_WEIGHTS.keys())
        if args.rule_based_only:
            allowed = [p for p in allowed if p in RULE_PATTERNS]
            print(f"Rule-based only mode. Patterns: {allowed}")
        else:
            print(f"Using patterns: {allowed}")

        if not args.rule_based_only and any(p in LLM_PATTERNS for p in allowed):
            client = create_llm_client(
                provider=args.provider, api_key=args.api_key, model=args.model,
            )
            print(f"{args.provider.capitalize()} client initialized. Model: {args.model}")

        # Load data (supports TSV and JSONL)
        records = load_input(args.input)
        if args.n:
            records = records[:args.n]
        print(f"Loaded {len(records)} records from {args.input}")

        # Generate
        output_records = []
        pattern_counts = {p: 0 for p in allowed}

        for i, record in enumerate(records):
            pattern = pick_pattern(allowed)
            pattern_counts[pattern] += 1

            is_llm = pattern in LLM_PATTERNS
            if i % 100 == 0:
                print(f"  [{i+1}/{len(records)}] pattern={pattern} {'(LLM)' if is_llm else '(rule)'}")

            noisy = generate_noisy(record, pattern, client=client)
            output_records.append(noisy)

            if is_llm and args.delay > 0:
                time.sleep(args.delay)

        save_jsonl(output_records, args.output)

        print(f"\n✓ Saved {len(output_records)} noisy records to {args.output}")
        print("\nPattern distribution:")
        for p, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            pct = count / len(output_records) * 100
            print(f"  {p:<25} {count:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
