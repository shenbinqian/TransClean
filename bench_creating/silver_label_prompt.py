"""
silver_label_prompt.py
======================
Prompt templates and response parsing for silver label generation.

Silver labels are extracted clean translations produced by multiple LLMs
(1 proprietary + 2 open-weight) and aggregated via majority vote to serve
as ground truth for extraction accuracy evaluation on curated data.

Unlike the method extraction prompt (llm_extractor.py), this prompt provides
additional context: source text, reference translation, language IDs, and
LLM-annotated noise patterns — enabling more reliable extraction.
"""

import json
import re


SILVER_LABEL_SYSTEM_PROMPT = """\
You are a translation quality expert. Your task is to extract the clean \
translation from a noisy machine translation output.

The noisy output may contain artifacts such as:
- Language prefixes or labels (e.g. "Chinese: ...")
- Verbose preambles (e.g. "Here is the translation...")
- Explanations, grammar notes, or word-by-word breakdowns
- Cultural notes or parenthetical comments
- Multiple alternative translations
- Bilingual output with both source and target text
- Code blocks, special formatting, or extra punctuation
- Wrong language or off-topic content

You will be given the source text, language pair, a reference translation, \
the noisy LLM output, and the identified noise patterns. Use all of this \
context to extract ONLY the clean translation in the target language.

If the output contains multiple translation alternatives, extract the best one.
If the output is entirely off-topic or in the wrong language, return an empty string.

Respond with ONLY valid JSON in this exact format:
{"extracted_translation": "the clean translation text only"}"""


SILVER_LABEL_USER_TEMPLATE = """\
Source ({src_lang} → {tgt_lang}):
{source}

Reference translation:
{reference}

Identified noise patterns: {noise_patterns}

Noisy LLM output to clean:
{translation}

Extract the clean translation. Respond with JSON only."""


def build_silver_label_prompt(record: dict) -> str:
    """Build the user message for silver label extraction from a curated record."""
    translation = record.get("translation", "")
    if len(translation) > 10000:
        translation = translation[:10000] + "\n[... truncated ...]"

    source = record.get("source", "")
    if len(source) > 5000:
        source = source[:5000] + " [... truncated ...]"

    reference = record.get("reference", "")
    if len(reference) > 5000:
        reference = reference[:5000] + " [... truncated ...]"

    noise_patterns = record.get("noise_patterns", [])
    if isinstance(noise_patterns, list):
        noise_patterns_str = ", ".join(noise_patterns) if noise_patterns else "unknown"
    else:
        noise_patterns_str = str(noise_patterns)

    return SILVER_LABEL_USER_TEMPLATE.format(
        src_lang=record.get("src_lang", "?"),
        tgt_lang=record.get("tgt_lang", "?"),
        source=source,
        reference=reference,
        noise_patterns=noise_patterns_str,
        translation=translation,
    )


def parse_silver_label_response(text: str) -> str | None:
    """Parse the extracted translation from an LLM response.

    Handles:
    - <think>...</think> blocks (Qwen3, DeepSeek)
    - Plain-text thinking headers
    - Markdown code fences around JSON
    - Raw JSON
    - Plain text fallback

    Returns the extracted translation string, or None if parsing fails.
    """
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip plain-text thinking blocks
    text = re.sub(
        r"(?i)^(thinking|reasoning|analysis):?\s*\n.*?\n\n(?=\{)",
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    # Try markdown code fences
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if "extracted_translation" in result:
                val = result["extracted_translation"]
                if isinstance(val, list):
                    return val[0] if val else None
                return str(val) if val is not None else None
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "extracted_translation" in result:
            val = result["extracted_translation"]
            if isinstance(val, list):
                return val[0] if val else None
            return str(val) if val is not None else None
    except (json.JSONDecodeError, TypeError):
        pass

    # Try finding JSON object in text
    json_match = re.search(
        r'\{[^{}]*"extracted_translation"[^{}]*\}', text, re.DOTALL
    )
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            val = result["extracted_translation"]
            if isinstance(val, list):
                return val[0] if val else None
            return str(val) if val is not None else None
        except (json.JSONDecodeError, KeyError):
            pass

    return None
