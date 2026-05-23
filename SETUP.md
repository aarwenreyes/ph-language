# Filipino Alpaca Dataset — Translation Setup Guide

This guide walks you through how to run `filipino_alpaca.py`, which translates the Waray Alpaca dataset into Filipino using the **Google Gemini API (free tier)**.

---

## Prerequisites

- Python 3.10 or higher
- A Google account (for the free API key)
- The dataset file: `alpaca_waray_clean.csv`

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

---

## Step 2 — Install Dependencies

```bash
pip install google-genai pandas tqdm
```

---

## Step 3 — Get a Free Gemini API Key

1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key — it looks like `AIzaSy...`

> The free tier gives you **1,000 requests/day** with no credit card required.

---

## Step 4 — Set Your API Key

### Windows (PowerShell) ✅
```powershell
$env:GEMINI_API_KEY="AIzaSy..."
```
> ⚠️ Do NOT use `set GEMINI_API_KEY=...` in PowerShell — it won't work. Use `$env:` instead.

### Mac / Linux
```bash
export GEMINI_API_KEY="AIzaSy..."
```

> This sets the key for the current terminal session only. You'll need to set it again if you open a new terminal window.

---

## Step 5 — Test with 10 Rows First

Before running the full dataset, do a quick test to make sure everything is working and the translations look correct.

**Windows (PowerShell):**
```powershell
python filipino_alpaca.py --input alpaca_waray_clean.csv --output alpaca_filipino_clean.csv --limit 10
```

**Mac / Linux:**
```bash
python filipino_alpaca.py --input alpaca_waray_clean.csv --output alpaca_filipino_clean.csv --limit 10
```

Open `alpaca_filipino_clean.csv` and check the `fil_instruction` and `fil_output` columns to verify the translations look correct.

---

## Step 6 — Run the Daily Batch

Once the test looks good, run the full translation. The script will automatically stop after ~990 rows to stay within the free daily limit.

```powershell
python filipino_alpaca.py --input alpaca_waray_clean.csv --output alpaca_filipino_clean.csv
```

**Run this command once per day.** The script saves progress to a checkpoint file (`checkpoint_waray_fil.jsonl`) and automatically resumes from where it left off each time.

---

## Progress & Output

After each run, the terminal will show a summary like this:

```
=======================================================
  Run complete!
  Rows translated this run : 990
  Errors (will retry)      : 0
  Total completed so far   : 990 / 51760
  Remaining                : 50770
  Estimated days left      : ~52
  Output CSV               : alpaca_filipino_clean.csv
  Checkpoint               : checkpoint_waray_fil.jsonl
=======================================================
```

The output CSV (`alpaca_filipino_clean.csv`) is updated after every run, so you have a usable partial dataset the whole time.

---

## Important Files

| File | Purpose |
|---|---|
| `filipino_alpaca.py` | The translation script |
| `alpaca_waray_clean.csv` | Input dataset (Waray) |
| `alpaca_filipino_clean.csv` | Output dataset (Waray + Filipino) |
| `checkpoint_waray_fil.jsonl` | Progress tracker — **do not delete this** |
| `translate_waray.log` | Log file for debugging errors |

> ⚠️ Never delete `checkpoint_waray_fil.jsonl`. It tracks which rows have already been translated. Deleting it means starting over from row 0.

---

## Troubleshooting

**`pyton` / `pyhon` is not recognized**
→ Typo. Make sure you type `python` exactly.

**`ValueError: Gemini API key not found`**
→ You're in PowerShell. Use `$env:GEMINI_API_KEY="..."` not `set GEMINI_API_KEY=...`

**`Missing expression after unary operator '--'`**
→ You're in PowerShell and used `\` to split the command across lines. Put the whole command on one line instead.

**Rate limit / 429 errors**
→ The script handles this automatically with retries. If it keeps happening, you've hit the daily limit — just run again tomorrow.

**Script stops early**
→ Normal behavior. It stops at ~990 rows to stay within the free daily limit. Run it again tomorrow to continue.
