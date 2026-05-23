"""
translate_waray_to_filipino.py
==============================
Translates the Waray Alpaca dataset (instruction, input, output columns)
into Filipino using the Google Gemini API (FREE tier).

Model used: gemini-2.5-flash-lite  (1,000 free requests/day, no credit card)

Each CSV row = 1 API call (all 3 fields batched into one prompt).
At 1,000 requests/day → ~52 days to translate all 51,760 rows.
Run the script daily; it resumes automatically from a checkpoint file.

Usage
-----
  # Test on 10 rows first
  python translate_waray_to_filipino.py \
    --input  alpaca_waray_clean.csv \
    --output alpaca_filipino_clean.csv \
    --limit  10

  # Daily production run (stops automatically at free-tier daily limit)
  python translate_waray_to_filipino.py \
    --input  alpaca_waray_clean.csv \
    --output alpaca_filipino_clean.csv

  # Resume after interruption (checkpoint is loaded automatically)
  python translate_waray_to_filipino.py \
    --input  alpaca_waray_clean.csv \
    --output alpaca_filipino_clean.csv

Requirements
------------
  pip install google-genai pandas tqdm

Environment variable
--------------------
  GEMINI_API_KEY  — your Google Gemini API key (required)
  Get one free at: https://aistudio.google.com/app/apikey
"""

import os
import re
import json
import time
import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL              = "gemini-2.5-flash-lite"   # highest free daily quota
MAX_OUTPUT_TOKENS  = 8192
REQUESTS_PER_MIN   = 12          # stay safely under the 15 RPM free limit
DAILY_LIMIT        = 990         # stop before hitting the 1,000 RPD hard cap
DEFAULT_CHECKPOINT = "checkpoint_waray_fil.jsonl"

SYSTEM_INSTRUCTION = (
    "You are an expert Filipino/Tagalog translator specializing in Philippine languages. "
    "Translate Waray (Winaray/Samar-Leyte Visayan) text into natural, fluent Filipino (Tagalog). "
    "Preserve meaning, tone, and all formatting exactly (numbered lists, bullet points, "
    "newlines, punctuation). Return ONLY the translated text, nothing else."
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("translate_waray.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Translation  — 1 API call per row (instruction + input + output batched)
# ─────────────────────────────────────────────────────────────────────────────

SEPARATOR = "===FIELD==="   # unique separator between fields in the prompt

def build_prompt(instruction: str, input_ctx: str, output: str) -> str:
    """
    Pack all three fields into a single prompt so we use only 1 API call per row.
    The model is asked to return them in the same separated format.
    """
    has_input = bool(input_ctx and input_ctx.strip())
    fields = [
        f"INSTRUCTION:\n{instruction}",
        f"INPUT:\n{input_ctx if has_input else '(empty)'}",
        f"OUTPUT:\n{output}",
    ]
    prompt = (
        "Translate each section below from Waray to Filipino. "
        f"Keep the section headers (INSTRUCTION, INPUT, OUTPUT) and the separator line ({SEPARATOR}) exactly as-is. "
        "Translate only the text content.\n\n"
        + f"\n{SEPARATOR}\n".join(fields)
    )
    return prompt


def parse_response(text: str) -> tuple[str, str, str]:
    """
    Parse the three translated fields back out of the model response.
    Falls back gracefully if the model doesn't follow the format exactly.
    """
    # Split on the separator
    parts = re.split(re.escape(SEPARATOR), text)

    def extract_field(block: str, label: str) -> str:
        # Remove the header line (e.g. "INSTRUCTION:") and strip whitespace
        cleaned = re.sub(rf"^\s*{label}\s*:\s*", "", block, flags=re.IGNORECASE).strip()
        if cleaned.lower() == "(empty)":
            return ""
        return cleaned

    if len(parts) >= 3:
        return (
            extract_field(parts[0], "INSTRUCTION"),
            extract_field(parts[1], "INPUT"),
            extract_field(parts[2], "OUTPUT"),
        )

    # Fallback: return whole text as instruction, empty for the rest
    log.warning("Unexpected response format — storing full text in fil_instruction")
    return text.strip(), "", ""


def translate_row(
    client: genai.Client,
    instruction: str,
    input_ctx: str,
    output: str,
    delay: float,
) -> tuple[str, str, str]:
    """
    Translate one CSV row. Returns (fil_instruction, fil_input, fil_output).
    Handles rate-limit (429) errors with exponential back-off.
    """
    if delay:
        time.sleep(delay)

    prompt = build_prompt(instruction, input_ctx, output)

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.1,   # low temperature = more consistent translations
                ),
            )
            return parse_response(response.text)

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 60 * (2 ** attempt)
                log.warning("Rate limit hit (attempt %d) — waiting %ds", attempt + 1, wait)
                time.sleep(wait)
            else:
                log.error("API error on attempt %d: %s", attempt + 1, e)
                if attempt == 4:
                    return (
                        f"[TRANSLATION ERROR: {type(e).__name__}]",
                        "",
                        f"[TRANSLATION ERROR: {type(e).__name__}]",
                    )
                time.sleep(5)

    return "[TRANSLATION ERROR: max retries]", "", "[TRANSLATION ERROR: max retries]"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    done = {}
    p = Path(path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done[rec["idx"]] = rec
            except json.JSONDecodeError:
                pass
    log.info("Loaded %d completed rows from checkpoint: %s", len(done), path)
    return done


def save_checkpoint_record(path: str, idx: int, fil_instruction: str,
                            fil_input: str, fil_output: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "idx":             idx,
            "fil_instruction": fil_instruction,
            "fil_input":       fil_input,
            "fil_output":      fil_output,
        }, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Translate Waray Alpaca dataset to Filipino using Gemini API (free tier)."
    )
    parser.add_argument("--input",      required=True,  help="Path to alpaca_waray_clean.csv")
    parser.add_argument("--output",     required=True,  help="Path for translated output CSV")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"JSONL checkpoint file (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Translate only this many rows (for testing)")
    parser.add_argument("--daily-limit", type=int, default=DAILY_LIMIT,
                        help=f"Max API calls per run (default: {DAILY_LIMIT} for free tier)")
    parser.add_argument("--api-key",    default=None,
                        help="Gemini API key (defaults to GEMINI_API_KEY env var)")
    args = parser.parse_args()

    # ── API client ────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found.\n"
            "Set it with:  export GEMINI_API_KEY=AIzaSyBF1Ywd_eboANyvuSFIBcOEAN-rUh2YIzM"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    client = genai.Client(api_key=api_key)

    # ── Load dataset ──────────────────────────────────────────────────────
    log.info("Loading dataset: %s", args.input)
    df = pd.read_csv(args.input)
    df["input"] = df["input"].fillna("")

    if args.limit:
        df = df.head(args.limit)
        log.info("Limiting to %d rows", args.limit)

    total_rows = len(df)
    log.info("Total rows in dataset: %d", total_rows)

    # ── Load checkpoint ───────────────────────────────────────────────────
    checkpoint = load_checkpoint(args.checkpoint)
    pending = [i for i in df.index if i not in checkpoint]
    log.info("%d rows pending | %d already done", len(pending), len(checkpoint))

    # ── Enforce daily limit ───────────────────────────────────────────────
    batch = pending[:args.daily_limit]
    if len(pending) > args.daily_limit:
        log.info(
            "Free tier daily limit: translating %d rows this run. "
            "Run again tomorrow for the next batch.",
            args.daily_limit,
        )

    # Delay between requests to stay under 15 RPM (60s / 12 req = 5s apart)
    delay = 60.0 / REQUESTS_PER_MIN

    # ── Translate ─────────────────────────────────────────────────────────
    errors = 0
    if batch:
        with tqdm(total=len(batch), desc="Translating", unit="rows") as pbar:
            for idx in batch:
                row = df.loc[idx]
                fil_instruction, fil_input, fil_output = translate_row(
                    client,
                    str(row["instruction"]),
                    str(row["input"]),
                    str(row["output"]),
                    delay,
                )

                if "[TRANSLATION ERROR" in fil_instruction or "[TRANSLATION ERROR" in fil_output:
                    errors += 1
                else:
                    # Only checkpoint successful translations so errors retry next run
                    save_checkpoint_record(
                        args.checkpoint, idx, fil_instruction, fil_input, fil_output
                    )
                    checkpoint[idx] = {
                        "idx":             idx,
                        "fil_instruction": fil_instruction,
                        "fil_input":       fil_input,
                        "fil_output":      fil_output,
                    }
                pbar.update(1)

    # ── Build output CSV from all completed rows ───────────────────────────
    log.info("Building output CSV from %d completed rows...", len(checkpoint))

    completed_df = df[df.index.isin(checkpoint)].copy()
    completed_df["fil_instruction"] = completed_df.index.map(
        lambda i: checkpoint[i]["fil_instruction"]
    )
    completed_df["fil_input"] = completed_df.index.map(
        lambda i: checkpoint[i]["fil_input"]
    )
    completed_df["fil_output"] = completed_df.index.map(
        lambda i: checkpoint[i]["fil_output"]
    )

    out_cols = [
        "language",
        "instruction",     "fil_instruction",
        "input",           "fil_input",
        "output",          "fil_output",
        "has_input",
    ]
    completed_df = completed_df[out_cols].copy()
    completed_df["language"] = "Waray-Filipino"

    completed_df.to_csv(args.output, index=False, encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────
    remaining = total_rows - len(checkpoint)
    days_left  = (remaining + args.daily_limit - 1) // args.daily_limit

    print("\n" + "=" * 55)
    print(f"  Run complete!")
    print(f"  Rows translated this run : {len(batch) - errors}")
    print(f"  Errors (will retry)      : {errors}")
    print(f"  Total completed so far   : {len(checkpoint)} / {total_rows}")
    print(f"  Remaining                : {remaining}")
    print(f"  Estimated days left      : ~{days_left}")
    print(f"  Output CSV               : {args.output}")
    print(f"  Checkpoint               : {args.checkpoint}")
    print("=" * 55)
    if remaining > 0:
        print("\n  Run this script again tomorrow to continue.\n")
    else:
        print("\n  Dataset fully translated!\n")


if __name__ == "__main__":
    main()