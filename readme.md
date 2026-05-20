# Philippine Language NLP Dataset & Tokenizer

Clean translation-pair datasets and an inference-ready tokenizer script for three Tagalog-based Philippine language pairs: **Waray**, **Hiligaynon**, and **Kapampangan**.

---

## Files

| File | Description |
|---|---|
| `translation_pairs.csv` | 1,656 translation pairs across all three language pairs |
| `alpaca_waray_clean.csv` | 51,760 Waray instruction-tuning records (Alpaca format) |
| `dataset_stats.json` | Machine-readable metadata for both CSVs |
| `tokenizer.py` | Tokenizer script for inference (rule-based + HuggingFace) |

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:
```bash
pip install pandas pyarrow transformers torch
```

> `transformers` and `torch` are only needed if you use the `--hf-model` flag. The rule-based tokenizer works with just `pandas`.

---

## Dataset Schemas

### `translation_pairs.csv`

| Column | Values |
|---|---|
| `language_pair` | `Tagalog-Waray`, `Tagalog-Hiligaynon`, `Tagalog-Kapampangan` |
| `split` | `train`, `test`, `vocab` |
| `source_lang` | Source language name |
| `target_lang` | Target language name |
| `source_text` | Source phrase/word |
| `target_text` | Target phrase/word |

**Row counts by split:**

| Language Pair | train | test | vocab |
|---|---|---|---|
| Tagalog-Waray | 497 | 11 | 764 |
| Tagalog-Hiligaynon | 172 | 20 | — |
| Tagalog-Kapampangan | 172 | 20 | — |

> `vocab` split contains word-level pairs from a separate JSONL source file — no train/test assignment was present in the original data.

### `alpaca_waray_clean.csv`

| Column | Description |
|---|---|
| `language` | Always `Waray` |
| `instruction` | Task instruction in Waray |
| `input` | Optional context (empty string if none) |
| `output` | Expected response in Waray |
| `has_input` | `True` if `input` is non-empty |

19,157 records include a context `input`; 32,603 are instruction-only.

---

## Tokenizer Usage

### Tokenize a single phrase
```bash
python tokenizer.py --text "Magandang umaga, kumusta ka na?"
```

### Tokenize translation pairs
```bash
# All pairs
python tokenizer.py --dataset translation_pairs.csv --mode translation

# Filter by language pair and split
python tokenizer.py --dataset translation_pairs.csv --mode translation \
  --pair Tagalog-Waray --split train

# Save output to JSONL
python tokenizer.py --dataset translation_pairs.csv --mode translation \
  --pair Tagalog-Waray --split train --output waray_train.jsonl
```

### Tokenize the Alpaca dataset
```bash
python tokenizer.py --dataset alpaca_waray_clean.csv --mode alpaca \
  --output alpaca_tokenized.jsonl
```

### Use a HuggingFace model
```bash
python tokenizer.py --dataset translation_pairs.csv --mode translation \
  --hf-model jcblaise/roberta-tagalog-base --output waray_hf.jsonl
```

Recommended models:
- [`jcblaise/roberta-tagalog-base`](https://huggingface.co/jcblaise/roberta-tagalog-base) — general Tagalog
- [`danjohnvelasco/bert-tagalog`](https://huggingface.co/danjohnvelasco/bert-tagalog) — Tagalog BERT

### All CLI flags

| Flag | Description |
|---|---|
| `--dataset` | Path to a CSV dataset file |
| `--mode` | `translation` or `alpaca` |
| `--text` | Tokenize a single string directly |
| `--pair` | Filter by language pair (translation mode) |
| `--split` | Filter by split: `train`, `test`, or `vocab` |
| `--output` | Save tokenized records to a JSONL file |
| `--hf-model` | HuggingFace model name for subword tokenization |
| `--no-lower` | Disable lowercasing |

---

## Tokenizer Output Format

Each tokenized record (JSONL) looks like this:

**Translation pair:**
```json
{
  "language_pair": "Tagalog-Waray",
  "split": "train",
  "source_lang": "Tagalog",
  "target_lang": "Waray",
  "source_text": "Istasyon ng Pulisya",
  "target_text": "Istasyon Hab Pulis",
  "source_tokens": ["istasyon", "ng", "pulisya"],
  "target_tokens": ["istasyon", "hab", "pulis"]
}
```

**Alpaca record:**
```json
{
  "language": "Waray",
  "instruction": "Paghatag hin tulo nga tip...",
  "input": "",
  "output": "1. Kaon hin balanse...",
  "has_input": false,
  "instruction_tokens": ["paghatag", "hin", "tulo", "nga", "tip"],
  "input_tokens": [],
  "output_tokens": ["1", ".", "kaon", "hin", "balanse", "..."]
}
```

---

## Data Cleaning Notes

The following issues were found and resolved during preprocessing:

- **Duplicate Waray parquet files** — the source zip contained two separate train+test parquet pairs. These were merged and deduplicated, reducing 914 → 497 unique train rows and 102 → 91 unique test rows.
- **Malformed JSONL record** — one record in `tagalog_to_waray.jsonl` had only one element in its `set` array (`babasagin -bubuungon`). It was skipped.
- **Inconsistent source column names** — Hiligaynon and Kapampangan used `Filipino` as the source column; Waray used `Tagalog`. Both are normalized to `source_lang`/`target_lang` in the merged CSV.