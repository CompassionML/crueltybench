#!/usr/bin/env python
"""Translate the non-English model responses in a ABCA eval log to English.

Reads a .eval log, and for every sample whose response is not in English, calls a
translator model (Opus 5 on AWS Bedrock by default) to produce a faithful English
translation. Writes a JSON file keyed by sample id — the exact shape make_report.py
consumes ({"<sample_id>": "<english text>", ...}) — so the report shows English for
non-English responses automatically.

Auth: needs the Bedrock bearer token + region in .env (AWS_BEARER_TOKEN_BEDROCK,
AWS_DEFAULT_REGION); this script loads .env itself. Judges/targets are unaffected.

Usage:
    uv run python scripts/translate_responses.py LOG.eval
    uv run python scripts/translate_responses.py LOG.eval --label claude-opus-5
    uv run python scripts/translate_responses.py LOG.eval --model bedrock/us.anthropic.claude-opus-5
    uv run python scripts/translate_responses.py LOG.eval --out path/to/translations.json

By default writes results/stats/translations_<slug>.json, where <slug> matches the
slug make_report.py derives from --label (or the log's model name) — so a subsequent
`make_report.py LOG.eval --label <same>` picks the file up with no extra flags.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import dotenv_values  # noqa: E402

# Load .env into the environment before importing/using the model provider, so the
# Bedrock bearer token + region are available (mirrors how Inspect auto-loads .env).
import os  # noqa: E402

for _k, _v in dotenv_values(REPO / ".env").items():
    if _v is not None:
        os.environ.setdefault(_k, _v)

from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402

STATS_DIR = REPO / "results" / "stats"
DEFAULT_MODEL = "bedrock/us.anthropic.claude-opus-5"

TRANSLATE_PROMPT = (
    "You are a professional translator. Translate the text below into natural, fluent English.\n"
    "- Preserve the meaning, tone, and register faithfully — do not soften, censor, or editorialise.\n"
    "- Preserve all Markdown formatting (headings, bold, bullet lists, numbering) exactly.\n"
    "- Do NOT add any preamble, notes, or explanation. Output ONLY the English translation.\n\n"
    "The text is written in {language}. Here it is:\n\n{text}"
)


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def short_model(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def final_answer(sample) -> str:
    if sample.output and sample.output.completion:
        return sample.output.completion
    for msg in reversed(sample.messages):
        if msg.role == "assistant" and getattr(msg, "text", ""):
            return msg.text
    return ""


async def translate_one(model, language: str, text: str) -> str:
    out = await model.generate(TRANSLATE_PROMPT.format(language=language, text=text))
    return out.completion.strip()


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log", help="Path to the .eval log to translate.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Translator model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--label", help="Slug label to match make_report.py (default: the log's model name).")
    ap.add_argument("--out", help="Output JSON path (default: results/stats/translations_<slug>.json).")
    ap.add_argument("--limit", type=int, help="Only translate the first N non-English responses (for testing).")
    args = ap.parse_args()

    log = read_eval_log(args.log)
    label = args.label or short_model(log.eval.model)
    slug = slugify(label)
    out_path = Path(args.out) if args.out else STATS_DIR / f"translations_{slug}.json"

    todo = []
    for s in log.samples or []:
        md = s.metadata or {}
        lang = (md.get("language") or "")
        if lang.lower() == "english":
            continue
        resp = final_answer(s)
        if not resp.strip():
            continue
        todo.append((s.id, lang, resp))

    if args.limit:
        todo = todo[: args.limit]

    print(f"Log:        {args.log}")
    print(f"Translator: {args.model}")
    print(f"To translate: {len(todo)} non-English response(s)")
    if not todo:
        print("Nothing to translate.")
        return

    # No temperature: the default translator (Opus 5) rejects it on Bedrock, and translation
    # doesn't need sampling control. max_tokens only.
    model = get_model(args.model, config=GenerateConfig(max_tokens=4096))

    translations: dict[str, str] = {}
    for i, (sid, lang, resp) in enumerate(todo, 1):
        try:
            en = await translate_one(model, lang, resp)
            translations[sid] = en
            print(f"  [{i}/{len(todo)}] {sid} [{lang}] -> {len(en)} chars")
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {sid} [{lang}] FAILED: {type(e).__name__}: {str(e)[:160]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {len(translations)} translations -> {out_path}")
    print(f"Next: uv run python scripts/make_report.py {args.log} --label {label}")


if __name__ == "__main__":
    asyncio.run(main())
