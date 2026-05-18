"""
prompts.py
==========
LLM prompt templates for noisy translation generation.

Contains system prompts, few-shot examples, and user templates for:
  - Noise patterns (explanation, cultural_note, bilingual_output, alternatives, verbose_preamble,
    wrong_language, off_topic)
"""

LANG_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "ru": "Russian",
    "he": "Hebrew",
    "cs": "Czech",
    "pl": "Polish",
    "ta": "Tamil",
    "km": "Khmer",
    "ko": "Korean",
    "it": "Italian",
    "es": "Spanish",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
}

# ═══════════════════════════════════════════════════════════════════
# NOISE GENERATION PROMPTS
# ═══════════════════════════════════════════════════════════════════
#
# These prompts are designed to produce realistic, content-aware noise
# that would be hard to generate with simple templates.
#
# Each prompt is a (system_prompt, user_prompt) pair.
# The user_prompt uses Python .format() placeholders:
#   {source}, {src_lang_name}, {tgt_lang_name}, {clean_translation}

SYSTEM_PROMPT = """\
You are simulating a large language model that generates translations with extra noise, \
explanations, or formatting artifacts. Your task is to take a clean reference translation \
and add realistic noise to it — exactly as a helpful-but-verbose LLM would.

Rules:
- Output ONLY the noisy translation (no meta-commentary, no "Here you go:", no JSON).
- The core translation meaning must remain correct.
- The noise must look authentic, as if a real LLM produced it.
- Use the target language for the translation itself; explanatory text may be in English \
or the source/target language depending on the pattern."""


LLM_PROMPT_TEMPLATES = {

    # ── Pattern 4: Newline-separated explanation ────────────────────────────────
    "explanation": {
        "description": (
            "Model translates correctly but then adds a word-by-word breakdown, "
            "romanization (Pinyin, transliteration), grammar notes, or cultural breakdown "
            "after a blank line. The first line is the clean translation."
        ),
        "few_shot_examples": [
            {
                "source": "This is one thing we need to change",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "这是我们需要改变的一点。",
                "noisy_output": (
                    "这是我们需要改变的一点。\n\n"
                    "(Zhè shì wǒmen xūyào gǎibiàn de yīdiǎn.)\n\n"
                    "**Explanation:**\n"
                    "* **这是 (Zhè shì):** This is\n"
                    "* **我们 (Wǒmen):** we\n"
                    "* **需要 (Xūyào):** need\n"
                    "* **改变 (Gǎibiàn):** change\n"
                    "* **的 (de):** (possessive particle)\n"
                    "* **一点 (yīdiǎn):** a bit / one thing"
                ),
            },
            {
                "source": "They wanted to introduce me and my company",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "他们想介绍我和我的公司。",
                "noisy_output": (
                    "他们想介绍我和我的公司。\n\n"
                    "(Tāmen xiǎng jièshào wǒ hé wǒ de gōngsī.)"
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation: {clean_translation}\n\n"
            "Generate the noisy output (translation + explanation/breakdown after a blank line):"
        ),
    },

    # ── Pattern 5: Cultural / contextual note ───────────────────────────────────
    "cultural_note": {
        "description": (
            "Model appends a parenthetical note explaining cultural nuances, idioms, "
            "or translation choices after the main translation."
        ),
        "few_shot_examples": [
            {
                "source": "Please listen to the Funeral Concerto",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "请听葬礼协奏曲。",
                "noisy_output": (
                    "请听葬礼协奏曲。\n\n"
                    "(Note: The 'Funeral Concerto' is not a specific well-known piece by a single "
                    "composer — this may refer to Tchaikovsky's Piano Concerto No. 1, sometimes "
                    "nicknamed as such. The translation preserves the literal meaning.)"
                ),
            },
            {
                "source": "We know you know someone in EKO, right?",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "我们知道你认识EKO里的人，是吗？",
                "noisy_output": (
                    "我们知道你认识EKO里的人，是吗？\n\n"
                    "（Note: 'EKO' is kept as-is since it appears to be a proper name or "
                    "organization. If it refers to a specific entity, the translation may need "
                    "to be adjusted accordingly.）"
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation: {clean_translation}\n\n"
            "Generate the translation with a cultural/contextual note appended in parentheses:"
        ),
    },

    # ── Pattern 6: Bilingual output ─────────────────────────────────────────────
    "bilingual_output": {
        "description": (
            "Model provides both source and target languages, often with language labels, "
            "meta-commentary, or a side-by-side format."
        ),
        "few_shot_examples": [
            {
                "source": "I love animals!!!",
                "src_lang_name": "Arabic", "tgt_lang_name": "Hebrew",
                "clean_translation": "אני אוהב חיות!!!",
                "noisy_output": (
                    "Arabic: أنا أحب الحيوانات!!!\n"
                    "Hebrew: אני אוהב חיות!!!\n"
                    "Translation: אני אוהב חיות!!!"
                ),
            },
            {
                "source": "laughing",
                "src_lang_name": "Arabic", "tgt_lang_name": "Chinese",
                "clean_translation": "笑",
                "noisy_output": (
                    "The Arabic word for 'laughing' is 'ضحك' (ḍaḥik). "
                    "In Chinese, this can be expressed as '笑 (xiào)'.\n\n"
                    "So, the translation would be: 笑"
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation: {clean_translation}\n\n"
            "Generate a bilingual noisy output that includes both languages "
            "with labels or meta-commentary:"
        ),
    },

    # ── Pattern 7: Alternative translations ────────────────────────────────────
    "alternatives": {
        "description": (
            "Model provides 2–4 numbered translation alternatives instead of a single output."
        ),
        "few_shot_examples": [
            {
                "source": "Eight peacekeeping missions",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "八项维和任务",
                "noisy_output": (
                    "Here are a few ways to translate this:\n\n"
                    "1. 八项维和任务 (literal)\n"
                    "2. 八次维和行动 (emphasizing 'operations')\n"
                    "3. 八个联合国维和任务 (adding 'UN' for clarity)"
                ),
            },
            {
                "source": "Vulnerable Dems air impeachment concerns to Pelosi",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "脆弱的民主党人就弹劾问题向佩洛西提出担忧",
                "noisy_output": (
                    "1. 脆弱的民主党人就弹劾问题向佩洛西表达担忧\n"
                    "2. 处境艰难的民主党人向佩洛西反映对弹劾的顾虑\n"
                    "3. 民主党内忧虑人士就弹劾议题向佩洛西进言"
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation (use as option 1): {clean_translation}\n\n"
            "Generate 2–3 alternative translations in a numbered list:"
        ),
    },

    # ── Pattern 2 (LLM): Verbose preamble ──────────────────────────────────────
    "verbose_preamble": {
        "description": (
            "Model writes an acknowledgement or context sentence before giving the translation, "
            "as if narrating the task."
        ),
        "few_shot_examples": [
            {
                "source": "Vulnerable Dems air impeachment concerns to Pelosi",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "脆弱的民主党人就弹劾问题向佩洛西提出担忧",
                "noisy_output": (
                    "Here is the translation of \"Vulnerable Dems air impeachment concerns "
                    "to Pelosi\" from English to Chinese:\n\n"
                    "脆弱的民主党人就弹劾问题向佩洛西提出担忧\n\n"
                    "This translates the headline in a way that preserves the political tone "
                    "of the original."
                ),
            },
            {
                "source": "Thomas Cook's descent into insolvency",
                "src_lang_name": "English", "tgt_lang_name": "German",
                "clean_translation": "Thomas Cooks Abstieg in die Insolvenz",
                "noisy_output": (
                    "Sure! Here is the German translation:\n\n"
                    "Thomas Cooks Abstieg in die Insolvenz"
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation: {clean_translation}\n\n"
            "Generate the output with a verbose preamble/acknowledgement before the translation:"
        ),
    },

    # ── Pattern 11: Wrong language ────────────────────────────────────
    "wrong_language": {
        "description": (
            "Model translates the source text but into the wrong language — "
            "e.g., the source language, English, or a related language — "
            "instead of the requested target language."
        ),
        "system_prompt_override": (
            "You are simulating a large language model that fails at the translation task. "
            "Your task is to produce output that does NOT correctly fulfill the translation request. "
            "Output ONLY the noisy result (no meta-commentary, no JSON)."
        ),
        "few_shot_examples": [
            {
                "source": "هذا الصباح انت حزين , تريد أن تستمع لاغنيتك,",
                "src_lang_name": "Arabic", "tgt_lang_name": "Chinese",
                "clean_translation": "今天早上你很伤心，你想听你的歌，",
                "noisy_output": (
                    "This morning you are sad, you want to listen to your song,"
                ),
            },
            {
                "source": "Eight peacekeeping missions",
                "src_lang_name": "English", "tgt_lang_name": "Chinese",
                "clean_translation": "八项维和任务",
                "noisy_output": (
                    "Huit missions de maintien de la paix"
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation (in {tgt_lang_name}): {clean_translation}\n\n"
            "Generate a translation of the source text, but in the WRONG language — "
            "use {src_lang_name}, English, or any language other than {tgt_lang_name}. "
            "Output only the wrong-language translation, nothing else:"
        ),
    },

    # ── Pattern 12: Off-topic ─────────────────────────────────────────
    "off_topic": {
        "description": (
            "Model completely ignores the translation task and produces irrelevant content — "
            "e.g., AI assistant boilerplate, code snippets, grammar lessons, "
            "unrelated Q&A, news text, or meta-commentary."
        ),
        "system_prompt_override": (
            "You are simulating a large language model that fails at the translation task. "
            "Your task is to produce output that does NOT correctly fulfill the translation request. "
            "Output ONLY the noisy result (no meta-commentary, no JSON)."
        ),
        "few_shot_examples": [
            {
                "source": "هذا الصباح انت حزين , تريد أن تستمع لاغنيتك,",
                "src_lang_name": "Arabic", "tgt_lang_name": "Chinese",
                "clean_translation": "今天早上你很伤心，你想听你的歌，",
                "noisy_output": (
                    "I am an AI assistant designed to be helpful. I can provide "
                    "information, answer questions, and help with tasks to the best "
                    "of my abilities. How can I help you today?"
                ),
            },
            {
                "source": "Thomas Cook's descent into insolvency",
                "src_lang_name": "English", "tgt_lang_name": "German",
                "clean_translation": "Thomas Cooks Abstieg in die Insolvenz",
                "noisy_output": (
                    "Here's a Python function to calculate compound interest:\n\n"
                    "```python\n"
                    "def compound_interest(principal, rate, time):\n"
                    "    return principal * (1 + rate) ** time\n"
                    "```\n\n"
                    "This function takes three parameters..."
                ),
            },
        ],
        "user_template": (
            "Source text: {source}\n"
            "Source language: {src_lang_name}\n"
            "Target language: {tgt_lang_name}\n"
            "Clean translation: {clean_translation}\n\n"
            "IGNORE the translation task above. Instead, generate completely unrelated content — "
            "for example: an AI assistant response, a code snippet, a grammar lesson, "
            "a news paragraph, or random Q&A. The output should have nothing to do with "
            "the source text or translation. Output only the off-topic content:"
        ),
    },
}



# ═══════════════════════════════════════════════════════════════════
# MESSAGE BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_llm_messages(pattern: str, source: str, src_lang: str, tgt_lang: str,
                        clean_translation: str) -> list[dict]:
    """
    Build the messages list for an LLM API call.
    Uses a few-shot format: system prompt + example pairs + the real request.
    """
    template = LLM_PROMPT_TEMPLATES[pattern]
    src_name = LANG_NAMES.get(src_lang, src_lang.upper())
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())

    messages = []

    # Inject few-shot examples as alternating user/assistant turns
    for ex in template["few_shot_examples"]:
        user_msg = template["user_template"].format(
            source=ex["source"],
            src_lang_name=ex["src_lang_name"],
            tgt_lang_name=ex["tgt_lang_name"],
            clean_translation=ex["clean_translation"],
        )
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ex["noisy_output"]})

    # Add the actual request
    user_msg = template["user_template"].format(
        source=source,
        src_lang_name=src_name,
        tgt_lang_name=tgt_name,
        clean_translation=clean_translation,
    )
    messages.append({"role": "user", "content": user_msg})

    return messages
