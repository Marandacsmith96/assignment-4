#!/usr/bin/env python3
"""Build the Assignment-4 notebook (MBAX 6418) as a valid .ipynb.

Every code cell is plain Python/jupyter code; the notebook runs top-to-bottom
in a Colab T4 runtime. Secrets are read from Colab userdata at runtime
(never embedded). Output: assignment4_finetune_comparison.ipynb

Cells are delimited with '''...''' so interior double quotes are safe.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assignment4_finetune_comparison.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

CELLS = []

# ----------------------------------------------------------------- title
CELLS.append(md('''# Assignment 4 — Fine-Tuning: Compare Three Approaches

**MBAX 6418 · Amazon Reviews '23 (Gift Cards) · POSITIVE / NEUTRAL / NEGATIVE**

One notebook compares **zero-shot prompting**, **few-shot prompting**, and
**fine-tuning (Unsloth + QLoRA)** on the same 300 test reviews, then explains
the results.

| Configuration (decided before running) | Value |
|---|---|
| Hosted model (zero-/few-shot) | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` at `http://dobolyi.com:9001/v1` (≈35B total, ~3B active, MoE) |
| Fine-tuned model | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (Llama-3.2-3B-Instruct) |
| QLoRA settings | r=16, alpha=32, dropout=0, all-linear targets, 4-bit, max_seq_len=1024 |
| Training settings | 3 epochs, lr 2e-4 (paged adamw 8-bit), batch 8, 10% warmup, linear schedule, seed 43 |
| Prompts | One shared task prompt; request exactly one label; temperature 0 (greedy), max 32 tokens, thinking disabled |
| Few-shot examples | 6 (2 per class) drawn ONLY from the prescribed training set |
| GPU requirement | T4 16 GB (training and inference) |

**Why this small model:** Llama-3.2-3B-Instruct is the best-documented
Unsloth fine-tuning model, is small enough for a T4 16 GB (4-bit QLoRA uses
≈6–8 GB during training), and its instruction quality is high for a 3B model,
giving the fine-tuned approach a fair chance against the 35B hosted model.

**Section plan:** data → checks → zero-shot → few-shot → fine-tune →
reloaded-adapter inference → evaluation → report → deliverables.'''))

# ----------------------------------------------------------------- runtime
CELLS.append(md('''## 0 · Runtime and package setup

Run this cell and confirm a **T4 GPU**. If `nvidia-smi` shows no GPU, pick
**Runtime → Change runtime type → T4 GPU** first. Cells below fail loudly if
the GPU is missing.'''))
CELLS.append(code('''import sys, time
print("python", sys.version.split()[0], "| colab:", "google.colab" in sys.modules)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU, then reconnect."
print("GPU OK — T4-class accelerator ready.")

!pip install -q --upgrade pip
!pip install -q unsloth "transformers" "peft" "trl" "accelerate" "bitsandbytes" "openai" "pandas" "scikit-learn" "scipy" "statsmodels" "matplotlib" "seaborn" "datasets"
import importlib
for m in ("unsloth", "openai", "sklearn", "statsmodels", "seaborn"):
    importlib.import_module(m)
print("all imports ok")'''))

# ----------------------------------------------------------------- credentials
CELLS.append(md('''## 1 · Credentials (never embedded in this notebook)

The hosted model needs the class API key. Add it once as a **Colab secret**:

1. Click the **key icon** (Secrets) in the left sidebar of Colab.
2. Add a secret named **`DOBOLYI_CLASS_KEY`** with the class key value.
3. Toggle "Notebook access" ON for this notebook.

The key is read into memory below and **never printed to the notebook**.'''))
CELLS.append(code('''import os
try:
    from google.colab import userdata
    KEY = userdata.get("DOBOLYI_CLASS_KEY")
except Exception:
    KEY = os.environ.get("DOBOLYI_CLASS_KEY", "")
if not KEY:
    raise RuntimeError("Missing DOBOLYI_CLASS_KEY secret. Add it (Secrets panel, notebook access ON), then re-run this cell.")
print("key loaded:", "yes" if KEY else "no", "(value hidden)")'''))

# ------------------------------------------------------------- drive checkpoints
CELLS.append(md('''## 1b · Google Drive checkpoints (disconnect protection)

If Colab ever drops the connection mid-run (laptop sleep, network blip,
browser update), the runtime restarts and everything in memory — and in
`/content` — is wiped. To survive that, every expensive stage saves its
finished results into a folder in **your Google Drive**. When you re-run,
finished stages are detected and skipped, and a half-finished evaluation
resumes exactly where it stopped.

Run this cell: it asks you to pick a Google account and click **Allow** (once
per notebook session).'''))
CELLS.append(code('''import os, csv
from pathlib import Path
from google.colab import drive
if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
CKPT_DIR = Path(os.environ.get("A4_CKPT_DIR", "/content/drive/MyDrive/assignment4_ckpts"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("checkpoint folder ready:", CKPT_DIR)

def stage_csv(name):
    return CKPT_DIR / ("predictions_" + name + ".csv")

def stage_part(name):
    return CKPT_DIR / ("predictions_" + name + ".part.csv")

def stage_complete(name, expected=300):
    """True when a finished result for this stage is already in Drive."""
    p = stage_csv(name)
    if not p.exists():
        return False
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) == expected

def _append_part(name, rec):
    """Append one finished review row to this stage's Drive progress file."""
    p = stage_part(name)
    fresh = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if fresh:
            w.writeheader()
        w.writerow(rec)'''))

# ----------------------------------------------------------------- data
CELLS.append(md('''## 2 · Fixed dataset and splits (do not change)

Source: Amazon Reviews '23, **Gift Cards**
(`review_categories/Gift_Cards.jsonl.gz`, McAuley lab).

Rules implemented exactly:
- 4–5 stars → **POSITIVE**; 3 → **NEUTRAL**; 1–2 → **NEGATIVE**
- `original_row` = original line number, starting at **1**
- Skip blank lines and reviews with fewer than **3 characters** of text after
  trimming (`text` = trimmed `title + " " + text`; missing text = `""`);
  unparseable lines and missing ratings are skipped too (counted below)
- `filtered_index` = 0-based rank of kept reviews
- **Train** = filtered_index 1,000–19,999 → `random.Random(43).sample(300)`
  per class, in class order POSITIVE, NEUTRAL, NEGATIVE, then the combined
  900 is shuffled with that same generator
- **Val** = filtered_index 20,000–29,999 → first 100 per class (reserved —
  **not used anywhere in this assignment**)
- **Test** = filtered_index 30,000–39,999 → first 100 per class (300 total;
  the SAME reviews for all three approaches)'''))
CELLS.append(code('''# download (12 MB)
import urllib.request, gzip, json
from pathlib import Path
DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
GZ = Path("/content/Gift_Cards.jsonl.gz")
if not GZ.exists():
    urllib.request.urlretrieve(DATA_URL, GZ)
print("dataset:", GZ, GZ.stat().st_size, "bytes")

# --- load + filter (assignment rules) ---
def label_of(rating: int) -> str:
    return "POSITIVE" if rating >= 4 else ("NEUTRAL" if rating == 3 else "NEGATIVE")

rows, skipped, short_examples = [], {"blank": 0, "json": 0, "no_rating": 0, "short": 0}, []
with gzip.open(GZ, "rt", encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        s = line.strip()
        if not s:
            skipped["blank"] += 1
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            skipped["json"] += 1
            continue
        try:
            rating = int(round(float(obj.get("rating"))))
        except (TypeError, ValueError):
            skipped["no_rating"] += 1
            continue
        title = obj.get("title") if isinstance(obj.get("title"), str) else ""
        text = obj.get("text") if isinstance(obj.get("text"), str) else ""
        content = f"{title} {text}".strip()
        if len(content) < 3:
            skipped["short"] += 1
            if len(short_examples) < 3:
                short_examples.append((lineno, content))
            continue
        rows.append({"original_row": lineno, "filtered_index": len(rows),
                     "rating": rating, "title": title, "text": text,
                     "content": content, "label": label_of(rating)})

print("filtered rows:", len(rows), " skipped:", skipped)
print("short-examples kept out:", short_examples)

# --- fixed splits ---
import random, collections
train_block, val_block, test_block = rows[1000:20000], rows[20000:30000], rows[30000:40000]
rng = random.Random(43)
train = []
for cls in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
    pool = [r for r in train_block if r["label"] == cls]
    train.extend(rng.sample(pool, 300))
rng.shuffle(train)

def first_per_class(pool, k=100):
    out, used = [], set()
    for cls in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        n = 0
        for r in pool:
            if r["label"] == cls and n < k:
                out.append(r)
                used.add(r["filtered_index"])
                n += 1
    return out

val, test = first_per_class(val_block), first_per_class(test_block)
print("train/val/test sizes:", len(train), len(val), len(test))
print("train classes:", collections.Counter(r["label"] for r in train))
print("val   classes:", collections.Counter(r["label"] for r in val))
print("test  classes:", collections.Counter(r["label"] for r in test))'''))

# ----------------------------------------------------------------- checks
CELLS.append(md('''## 3 · Required data checks (fail = stop)

Checks per the assignment: set sizes, class counts, unique IDs within each
set, **no shared IDs across sets**, exactly 900 training records with 300 per
class, few-shot examples drawn from the training set, and **only review titles
+ text used as inputs** (never the rating). Any failed check raises and stops
this notebook.'''))
CELLS.append(code('''import collections
assert len(train) == 900, "train != 900"
assert collections.Counter(r["label"] for r in train) == collections.Counter(
    {"POSITIVE": 300, "NEUTRAL": 300, "NEGATIVE": 300}), "train not 300/class"
assert len(val) == 300 and len(test) == 300
for s in (val, test):
    assert collections.Counter(r["label"] for r in s) == collections.Counter(
        {"POSITIVE": 100, "NEUTRAL": 100, "NEGATIVE": 100}), "val/test not 100/class"
t_ids, v_ids, e_ids = ({r["filtered_index"] for r in s} for s in (train, val, test))
assert len(t_ids) == 900 and len(v_ids) == 300 and len(e_ids) == 300, "duplicate IDs"
assert not (t_ids & v_ids) and not (t_ids & e_ids) and not (v_ids & e_ids), "ID overlap!"
assert all(set(r) >= {"title", "text", "content", "label"} for r in train + val + test)
print("ALL DATA CHECKS PASSED")

import csv
with open("/content/train_ids.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filtered_index", "original_row", "label", "rating"])
    for r in train:
        w.writerow([r["filtered_index"], r["original_row"], r["label"], r["rating"]])
with open("/content/test_ids.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filtered_index", "original_row", "label", "rating"])
    for r in test:
        w.writerow([r["filtered_index"], r["original_row"], r["label"], r["rating"]])
print("wrote /content/train_ids.csv and /content/test_ids.csv")'''))

# ----------------------------------------------------------------- formatted examples
CELLS.append(md('''## 4 · Formatted training examples (review before training)

Five real training rows, formatted exactly as they will be fed to the model
(same structure at training and inference). Star-rating labels may not match
every nuance of the text — that is expected and is part of the analysis.'''))
CELLS.append(code('''def fmt(row):
    return (f"### TRAINING EXAMPLE (filtered_index={row['filtered_index']}, "
            f"reference={row['label']})\\n"
            f"Review title: {row['title']}\\n"
            f"Review text: {row['text']}\\n"
            f"Expected label: {row['label']}\\n")
for i in range(5):
    print(fmt(train[i]))
print("(500 more training examples follow the same format)")'''))

# ----------------------------------------------------------------- prompt config
CELLS.append(md('''## 5 · Shared prompt & generation settings

One task instruction is used by **all three approaches**. Generation settings
are recorded and identical in spirit: **temperature 0 (greedy)**, small
output-token limit (32), **thinking disabled** (the hosted model is a
reasoning model; without `enable_thinking:false` it spends the token budget
on its reasoning chain and returns an empty answer — verified before
building this notebook; the flag is sent via `extra_body` so the current
openai SDK transmits it reliably).

Few-shot examples (2 per class, from the shuffled **training set**):
POSITIVE: filtered_index 2325, 18906 · NEUTRAL: 3501, 9216 · NEGATIVE: 4969, 14785'''))
CELLS.append(code('''# --- shared configuration ---
SYSTEM_PROMPT = ("You are a sentiment classifier for Amazon product reviews. "
                 "Classify each review as exactly one of: POSITIVE, NEUTRAL, "
                 "NEGATIVE. Reply with only the label and nothing else.")
USER_TEMPLATE = "Review title: {title}\\nReview text: {text}\\nSentiment:"
GEN = {"temperature": 0.0, "max_tokens": 32, "enable_thinking": False}
HOSTED_BASE = "http://dobolyi.com:9001/v1"
HOSTED_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

FEWSHOT = [  # (filtered_index, label, title, text) — all from train (asserted)
    (2325,  "POSITIVE", "I love these gift boxes!",
     "I love the fact that you can just order these and get them as a fast last minute gift"),
    (18906, "POSITIVE", "Everyone Loves Movies!",
     "Great gift which ranks right up there with general Amazon Gift Cards.  Everyone loves movies!"),
    (3501,  "NEUTRAL",  "Cash vs card", "High purchase fee."),
    (9216,  "NEUTRAL",  "Oversized, will not fit in a standard thank you card",
     "This is so cute, but too big to fit in a thank you card and mail."),
    (4969,  "NEGATIVE", "Pretty design but useless",
     "The Gift Card had no denomination printed on it.  There was no envelope enclosed."),
    (14785, "NEGATIVE", "Gift Card was torn.",
     "Card was ripped almost in half and scrunched into the box."),
]
assert all(fi in t_ids and lbl == next(r["label"] for r in train if r["filtered_index"] == fi)
           for fi, lbl, _, _ in FEWSHOT), "few-shot examples must come from the training set"
print("FEW-SHOT EXAMPLES VERIFIED: all 6 drawn from the prescribed training set")
print("prompt system:", SYSTEM_PROMPT)
print("prompt user:  ", USER_TEMPLATE)
print("generation:", GEN)'''))

# ----------------------------------------------------------------- helpers
CELLS.append(md('''## 6 · Scoring helpers

Label normalization is exactly as specified: **trim whitespace, normalize
case, accept only POSITIVE / NEUTRAL / NEGATIVE**; anything else is recorded
as **INVALID** (never discarded, never guessed). API failures are retried
with backoff; if all retries fail the raw response field records the error
and the prediction is INVALID — a real count, not a dropped row.

All responses are **cached to disk** (prompt + model + review id), so
re-running an approach never silently re-calls the API.'''))
CELLS.append(code('''import time, json
from pathlib import Path
from openai import OpenAI

CACHE_DIR = Path("/content/scoring_cache")
CACHE_DIR.mkdir(exist_ok=True)
client = OpenAI(base_url=HOSTED_BASE, api_key=KEY)

def normalize_label(raw):
    # Trim, normalize case/punctuation; only the 3 labels pass; else INVALID.
    if raw is None:
        return "INVALID"
    s = "".join(ch for ch in str(raw).upper() if ch.isalpha())
    return s if s in ("POSITIVE", "NEUTRAL", "NEGATIVE") else "INVALID"

def _cache_path(approach):
    return CACHE_DIR / (approach + ".jsonl")

def _load_cache(approach):
    d = {}
    p = _cache_path(approach)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                d[e["filtered_index"]] = e
    return d

def classify_hosted(messages, approach, fi, retries=3):
    # Cached hosted-model call. Returns (prediction, raw_response, cached_flag).
    # Cached INVALID entries are NOT trusted: they may come from a broken
    # transport, so they are re-scored (with retries) instead of reused.
    cache = _load_cache(approach)
    if fi in cache and cache[fi]["prediction"] != "INVALID":
        e = cache[fi]
        return e["prediction"], e["raw_response"], True
    raw, err = "", ""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=HOSTED_MODEL, messages=messages,
                temperature=GEN["temperature"], max_tokens=GEN["max_tokens"],
                # disable the reasoning chain; extra_body is the SDK-agnostic
                # way to pass chat_template_kwargs (openai >=2 drops the old
                # direct kwarg, which made the model think and return empty)
                extra_body={"chat_template_kwargs": {"enable_thinking": GEN["enable_thinking"]}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            err = "api_error(attempt %d): %s" % (attempt, type(e).__name__)
            time.sleep(2 ** attempt)
    if not raw and err:
        raw = err
    pred = normalize_label(raw)
    with _cache_path(approach).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"filtered_index": fi, "model": HOSTED_MODEL,
                            "prompt": messages, "prediction": pred,
                            "raw_response": raw, "ts": time.time()}) + "\\n")
    return pred, raw, False

def review_messages(row, fewshot=False):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    user = ""
    if fewshot:
        for _, lbl, t, x in FEWSHOT:
            user += USER_TEMPLATE.format(title=t, text=x) + " " + lbl + "\\n\\n"
    user += USER_TEMPLATE.format(title=row["title"], text=row["text"])
    msgs.append({"role": "user", "content": user})
    return msgs

print("helpers ready; cache dir:", CACHE_DIR)

# --- one-call sanity probe: shows the raw model answer + its normalized label
#     BEFORE the 300-review loop, so a broken transport is visible instantly
try:
    probe_msgs = review_messages(test[0], fewshot=False)
    probe_resp = client.chat.completions.create(
        model=HOSTED_MODEL, messages=probe_msgs,
        temperature=GEN["temperature"], max_tokens=GEN["max_tokens"],
        extra_body={"chat_template_kwargs": {"enable_thinking": GEN["enable_thinking"]}},
    )
    probe_raw = (probe_resp.choices[0].message.content or "").strip()
    print("probe review %d: raw=%r -> %s"
          % (test[0]["filtered_index"], probe_raw[:80], normalize_label(probe_raw)))
except Exception as e:
    print("probe call failed (continuing anyway):", type(e).__name__, str(e)[:120])'''))

# ----------------------------------------------------------------- zero-shot
CELLS.append(md('''## 7 · Approach A — ZERO-SHOT (hosted model)

All 300 test reviews, same prompt, one label requested. Outputs are written
to `/content/predictions_zero_shot.csv` (same schema as the submitted file:
`original_row, filtered_index, title, text, reference_label, prediction,
raw_response`) and mirrored to the Drive checkpoint folder. Every finished
review is saved to Drive as it is scored, so a disconnected run **resumes**
when you re-run; a fully finished stage is **skipped** entirely.'''))
CELLS.append(code('''import json, csv, shutil, time

def run_approach(name, fewshot):
    start = time.time()
    # 1) already finished? reuse the Drive copy (and validate against the prescribed IDs)
    if stage_complete(name):
        shutil.copyfile(stage_csv(name), "/content/predictions_%s.csv" % name)
        with open("/content/predictions_%s.csv" % name, newline="", encoding="utf-8") as f:
            out = list(csv.DictReader(f))
        ids_ok = {int(r["filtered_index"]) for r in out} == e_ids
        all_invalid = len(out) > 0 and all(r["prediction"] == "INVALID" for r in out)
        if ids_ok and not all_invalid:
            invalids = sum(r["prediction"] == "INVALID" for r in out)
            print("  [%s] REUSED saved Drive results (%d rows, %d invalid) - stage skipped" % (name, len(out), invalids))
            return out
        if all_invalid:
            print("  [%s] Drive results are ALL INVALID (broken earlier run) - re-scoring from scratch" % name)
        else:
            print("  [%s] Drive copy does not match the prescribed test IDs - re-running" % name)

    # 2) resume a half-finished stage (per-review progress is already in Drive);
    #    INVALID rows are re-scored, so a broken earlier run can never poison a rerun
    done = {}
    if stage_part(name).exists():
        with open(stage_part(name), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[int(r["filtered_index"])] = r
        n_bad = sum(1 for r in done.values() if r["prediction"] == "INVALID")
        if done:
            print("  [%s] resuming: %d of 300 reviewed (%d invalid rows will be re-scored)"
                  % (name, len(done), n_bad))

    # 3) score the missing reviews; append each finished row to Drive immediately
    out, invalids = [], 0
    for i, row in enumerate(test, 1):
        prev = done.get(row["filtered_index"])
        if prev is not None and prev["prediction"] != "INVALID":
            out.append(prev)
            continue
        msgs = review_messages(row, fewshot=fewshot)
        pred, raw, cached = classify_hosted(msgs, name, row["filtered_index"])
        invalids += (pred == "INVALID")
        rec = {"original_row": row["original_row"],
               "filtered_index": row["filtered_index"],
               "title": row["title"], "text": row["text"],
               "reference_label": row["label"],
               "prediction": pred, "raw_response": raw}
        out.append(rec)
        _append_part(name, rec)
        if i % 50 == 0 or i == 300:
            print("  [%s] %d/300  invalid-so-far: %d" % (name, i, invalids), flush=True)

    # 4) finalize: local CSV (as before) + durable Drive copy; clear the part file
    path = "/content/predictions_%s.csv" % name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    shutil.copyfile(path, stage_csv(name))
    if stage_part(name).exists():
        stage_part(name).unlink()
    print("  [%s] saved %s + Drive copy (%d rows, %d invalid) in %.0fs" % (name, path, len(out), invalids, time.time() - start))
    return out

t0 = time.time()
zero_shot = run_approach("zero_shot", fewshot=False)
print("zero-shot done in %.1fs" % (time.time() - t0))'''))

# ----------------------------------------------------------------- few-shot
CELLS.append(md('''## 8 · Approach B — FEW-SHOT (hosted model + 6 training examples)

Identical prompt and settings; the 6 verified training examples (2 per
class) are prepended inside the user message before the target review.
Same Drive checkpoint behavior as Approach A: resumes if half-done, skips
if already finished.'''))
CELLS.append(code('''t0 = time.time()
few_shot = run_approach("few_shot", fewshot=True)
print("few-shot done in %.1fs" % (time.time() - t0))'''))

# ----------------------------------------------------------------- fine-tune setup
CELLS.append(md('''## 9 · Approach C — FINE-TUNING (Unsloth + QLoRA, Llama-3.2-3B)

Installs Unsloth, loads `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` in 4-bit,
adds LoRA (r=16, alpha=32, all-linear), trains 3 epochs on the 900
prescribed reviews (batch 8, lr 2e-4, warmup 10%, seed 43), and saves ONLY
the adapter. Training fits a T4 16 GB (≈6–8 GB VRAM).

The training records are formatted with the **same system prompt + user
template** used at inference; the assistant turn is the label, so the model
learns to answer with exactly one label.

**Disconnect protection:** training checkpoints go to Drive every 50 steps,
so an interrupted training **resumes**; if the adapter ZIP is already in
Drive, this setup and the training below are **skipped** entirely.'''))
CELLS.append(code('''FINAL_ZIP = CKPT_DIR / "assignment4_adapter.zip"
TRAINED = FINAL_ZIP.exists() and (CKPT_DIR / "lora_model").exists()
BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
if TRAINED:
    print("fine-tuning already finished (adapter ZIP found in Drive) - setup skipped")
else:
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID, max_seq_length=1024, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=43,
    )
    print("base model loaded:", BASE_MODEL_ID)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", trainable)'''))
CELLS.append(code('''if TRAINED:
    print("fine-tuning already finished - training skipped")
else:
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported
    import shutil

    def to_chat(row):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(title=row["title"], text=row["text"])},
                {"role": "assistant", "content": row["label"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    from datasets import Dataset
    train_ds = Dataset.from_list([to_chat(r) for r in train])
    print("formatted training records:", len(train_ds))
    for i in range(3):
        print("---", train_ds[i]["text"][-160:].replace("\\n", " | "))

    # NOTE (transformers >= 5): warmup_ratio was renamed to warmup_steps,
    # which accepts a fraction (0.1 == 10% warmup, unchanged methodology).
    TRAIN_KW = dict(
        per_device_train_batch_size=8, gradient_accumulation_steps=1,
        warmup_steps=0.1, num_train_epochs=3, learning_rate=2e-4,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=43,
        output_dir=str(CKPT_DIR / "ft_outputs"),
        report_to="none",
        save_strategy="steps", save_steps=50, save_total_limit=2,
    )
    try:
        args = TrainingArguments(**TRAIN_KW)
    except TypeError as e:
        # Only transformers >= 5 is supported by this cell; fail loudly
        # instead of guessing across API generations.
        raise RuntimeError("unsupported transformers API for TrainingArguments: %s" % e)

    # Build the trainer; tolerate both the classic trl kwargs and the modern
    # SFTConfig-style API, so the training step survives either version.
    try:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds,
                             dataset_text_field="text", max_seq_length=1024, args=args)
    except TypeError:
        from trl import SFTConfig
        sft_args = SFTConfig(**{**TRAIN_KW, "dataset_text_field": "text", "max_seq_length": 1024})
        trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds)
    ckpts = sorted((CKPT_DIR / "ft_outputs").glob("checkpoint-*"))
    if ckpts:
        print("resuming training from", ckpts[-1])
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    model.save_pretrained("/content/lora_model")
    shutil.copytree("/content/lora_model", CKPT_DIR / "lora_model", dirs_exist_ok=True)
    print("adapter saved to /content/lora_model + Drive backup")'''))

# ----------------------------------------------------------------- adapter zip
CELLS.append(md('''## 10 · Adapter ZIP (exactly two files)

The submitted ZIP contains **only** `adapter_model.safetensors` and
`adapter_config.json`. The config identifies the original base model
(`base_model_name_or_path` points at the 4-bit
`unsloth/Llama-3.2-3B-Instruct-bnb-4bit` checkpoint, which maps to Meta's
Llama-3.2-3B-Instruct; the reload cell below documents this chain).'''))
CELLS.append(code('''import zipfile, shutil
ADAPTER_DIR = Path("/content/lora_model")
ZIP_PATH = Path("/content/assignment4_adapter.zip")
if not ADAPTER_DIR.exists() or not (ADAPTER_DIR / "adapter_model.safetensors").exists():
    # /content was wiped (runtime restart): restore the adapter from Drive
    src = CKPT_DIR / "lora_model"
    assert src.exists(), "no adapter anywhere: Drive backup missing; re-run the training cell"
    ADAPTER_DIR.mkdir(exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(src / name, ADAPTER_DIR / name)
    print("adapter restored from Drive backup")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        p = ADAPTER_DIR / name
        assert p.exists(), "missing " + name
        z.write(p, name)
shutil.copyfile(ZIP_PATH, FINAL_ZIP)
print("adapter zip:", ZIP_PATH, ZIP_PATH.stat().st_size, "bytes (+ Drive backup)")
with zipfile.ZipFile(ZIP_PATH) as z:
    print("contents:", z.namelist())'''))

# ----------------------------------------------------------------- reload + inference example
CELLS.append(md('''## 11 · Reload the SUBMITTED adapter — complete inference example

This is the required "saved-adapter inference" section:

- Extracts the **submission ZIP** into a fresh folder
- Loads a **fresh model instance** of the exact base model + matching tokenizer
- Applies the evaluation prompt and the same generation settings
- Classifies reviews from title + text **without training and without hosted-model calls**
- Shows the **raw response and the final label** (or INVALID)

Recorded below: model and tokenizer identifiers, tested package versions, and
generation settings — all printed from the actual environment.'''))
CELLS.append(code('''import zipfile, json
import importlib.metadata as md

EXTRACT = Path("/content/adapter_reloaded")
EXTRACT.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall(EXTRACT)
zf = json.loads((EXTRACT / "adapter_config.json").read_text())
print("adapter_config.base_model_name_or_path:", zf.get("base_model_name_or_path"))

model2, tokenizer2 = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID, max_seq_length=1024, dtype=None, load_in_4bit=True,
)
model2 = FastLanguageModel.for_inference(model2)
model2.load_adapter(str(EXTRACT))
print("adapter reloaded from the submitted ZIP onto a FRESH base model instance")
print("tokenizer id:", getattr(tokenizer2, "name_or_path", "?"))
print("model id:    ", getattr(model2, "name_or_path", "?"))
for pkg in ("torch", "transformers", "unsloth", "peft", "trl", "openai", "sklearn", "statsmodels"):
    try:
        print("  %s == %s" % (pkg, md.version(pkg)))
    except Exception:
        pass
print("generation settings:", GEN, "| do_sample=False (greedy), max_new_tokens=32")

for row in (train[12], train[25], train[7]):
    msgs = review_messages(row, fewshot=False)
    tokens = tokenizer2.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model2.generate(input_ids=tokens, max_new_tokens=32, do_sample=False, temperature=0.0)
    raw = tokenizer2.decode(out[0][tokens.shape[1]:], skip_special_tokens=True).strip()
    pred = normalize_label(raw)
    print("filtered_index=%d reference=%s -> raw=%r prediction=%s"
          % (row["filtered_index"], row["label"], raw, pred))'''))

# ----------------------------------------------------------------- ft predictions + prompt examples
CELLS.append(md('''## 12 · Fine-tuned test predictions + prompt examples

The **final prediction file** and `prompt_examples.jsonl` are generated with
the **reloaded adapter** (the exact submitted artifact), using the same
procedure as the inference example above — no training, no hosted-model calls.

Like the hosted stages, results are mirrored to the Drive checkpoint folder
every review, so this stage **skips** if already finished and **resumes**
if half-done when you re-run.'''))
CELLS.append(code('''import json, shutil
if stage_complete("fine_tuned"):
    shutil.copyfile(stage_csv("fine_tuned"), "/content/predictions_fine_tuned.csv")
    ft_out = []
    with open("/content/predictions_fine_tuned.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ft_out.append(r)
    all_invalid = len(ft_out) > 0 and all(r["prediction"] == "INVALID" for r in ft_out)
else:
    ft_out, all_invalid = [], False
if stage_complete("fine_tuned") and not all_invalid:
    pe_src = CKPT_DIR / "prompt_examples.jsonl"
    assert pe_src.exists(), "predictions found in Drive but prompt_examples.jsonl is missing"
    shutil.copyfile(pe_src, "/content/prompt_examples.jsonl")
    invalids = sum(r["prediction"] == "INVALID" for r in ft_out)
    print("REUSED saved Drive fine-tuned results (%d rows, %d invalid) - stage skipped" % (len(ft_out), invalids))
else:
    if all_invalid:
        print("saved fine-tuned Drive results are ALL INVALID (broken earlier run) - re-scoring from scratch")
    # resume a half-finished pass (per-review progress is already in Drive);
    # INVALID rows are re-scored so a broken earlier run can't poison a rerun
    done = {}
    if stage_part("fine_tuned").exists():
        with open(stage_part("fine_tuned"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[int(r["filtered_index"])] = r
        n_bad = sum(1 for r in done.values() if r["prediction"] == "INVALID")
        if done:
            print("resuming fine-tuned scoring: %d of 300 done (%d invalid rows will be re-scored)"
                  % (len(done), n_bad))

    saved_prompt_classes = set()
    pe_path = CKPT_DIR / "prompt_examples.jsonl"
    if pe_path.exists():
        for line in pe_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                saved_prompt_classes.add(json.loads(line)["reference_label"])

    ft_out, invalids = [], 0
    with open(pe_path, "a", encoding="utf-8") as pex:
        for row in test:
            prev = done.get(row["filtered_index"])
            if prev is not None and prev["prediction"] != "INVALID":
                ft_out.append(prev)
                continue
            msgs = review_messages(row, fewshot=False)
            tokens = tokenizer2.apply_chat_template(msgs, tokenize=True,
                                                    add_generation_prompt=True, return_tensors="pt").to("cuda")
            out = model2.generate(input_ids=tokens, max_new_tokens=32, do_sample=False, temperature=0.0)
            raw = tokenizer2.decode(out[0][tokens.shape[1]:], skip_special_tokens=True).strip()
            pred = normalize_label(raw)
            invalids += (pred == "INVALID")
            rec = {"original_row": row["original_row"],
                   "filtered_index": row["filtered_index"],
                   "title": row["title"], "text": row["text"],
                   "reference_label": row["label"],
                   "prediction": pred, "raw_response": raw}
            ft_out.append(rec)
            _append_part("fine_tuned", rec)
            if row["label"] not in saved_prompt_classes:
                pex.write(json.dumps({"original_row": row["original_row"],
                                      "filtered_index": row["filtered_index"],
                                      "reference_label": row["label"],
                                      "messages": msgs}) + "\\n")
                saved_prompt_classes.add(row["label"])
            del tokens, out
            if len(ft_out) % 50 == 0:
                print("  %d/300" % len(ft_out), flush=True)
        assert len(saved_prompt_classes) == 3, "need one prompt example per class"
    with open("/content/predictions_fine_tuned.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ft_out[0].keys()))
        w.writeheader()
        w.writerows(ft_out)
    shutil.copyfile("/content/predictions_fine_tuned.csv", stage_csv("fine_tuned"))
    shutil.copyfile(pe_path, "/content/prompt_examples.jsonl")  # deliverables cell reads it from /content
    if stage_part("fine_tuned").exists():
        stage_part("fine_tuned").unlink()
    print("wrote /content/predictions_fine_tuned.csv (300 rows, %d invalid) + Drive copy" % invalids)
    print("wrote", pe_path, "(3 entries)")'''))

# ----------------------------------------------------------------- metrics
CELLS.append(md('''## 13 · Evaluation — metrics and per-class detail

Balanced accuracy and macro-F1 for each approach, plus per-class
precision/recall/F1 with support, and correct/incorrect counts. Metrics are
averaged over the three sentiment classes; INVALID is a *prediction* state
(counted as incorrect), never a fourth reference class.'''))
CELLS.append(code('''import pandas as pd, numpy as np, json
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

def load_pred(path):
    return pd.read_csv(path)

frames = {"zero-shot": load_pred("/content/predictions_zero_shot.csv"),
          "few-shot": load_pred("/content/predictions_few_shot.csv"),
          "fine-tuned": load_pred("/content/predictions_fine_tuned.csv")}
for name, df in frames.items():
    assert len(df) == 300 and df["filtered_index"].nunique() == 300, name + " not 300 unique"
    assert set(df["filtered_index"]) == e_ids, name + " IDs != prescribed test IDs"
print("ID checks: each approach has exactly one prediction per prescribed test ID (aligned by ID)")

summary = {}
for name, df in frames.items():
    y_true, y_pred = df["reference_label"], df["prediction"]
    bal = balanced_accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    n_invalid = int((y_pred == "INVALID").sum())
    correct = int((y_true == y_pred).sum())
    summary[name] = {"balanced_acc": round(bal, 4), "macro_f1": round(macro, 4),
                     "correct": correct, "incorrect": 300 - correct, "invalid": n_invalid,
                     "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4),
                                       "f1": round(f[i], 4), "support": int(s[i])}
                                   for i, c in enumerate(CLASSES)}}
for name, m in summary.items():
    print("=== %s === bal_acc=%s macro_f1=%s correct=%d incorrect=%d invalid=%d"
          % (name, m["balanced_acc"], m["macro_f1"], m["correct"], m["incorrect"], m["invalid"]))
    for c, v in m["per_class"].items():
        print("   %-9s P=%s R=%s F1=%s support=%s" % (c, v["precision"], v["recall"], v["f1"], v["support"]))
with open("/content/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)'''))

# ----------------------------------------------------------------- confusion matrices
CELLS.append(md('''## 14 · Confusion matrices (reference rows → prediction columns)

Three matrices **side by side**, row-normalized (each reference row sums to
1), plus an **enlarged matrix for the approach with the highest macro-F1**.
If any INVALID predictions occur, an INVALID column appears in that matrix.'''))
CELLS.append(code('''import matplotlib.pyplot as plt

def row_normalized_cm(df):
    y_true, y_pred = df["reference_label"], df["prediction"]
    cols = CLASSES + (["INVALID"] if (y_pred == "INVALID").any() else [])
    cm = confusion_matrix(y_true, y_pred, labels=cols)
    return cm.astype(float) / cm.sum(axis=1, keepdims=True), cols

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, df) in zip(axes, frames.items()):
    cm, cols = row_normalized_cm(df)
    ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_title("%s  (macro-F1 %s)" % (name, summary[name]["macro_f1"]))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_xlabel("prediction"); ax.set_ylabel("reference")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "%.2f" % cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
plt.tight_layout()
plt.savefig("/content/confusion_matrices_3x.png", dpi=150)
plt.show()

best = max(summary, key=lambda k: summary[k]["macro_f1"])
cm, cols = row_normalized_cm(frames[best])
fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
ax2.imshow(cm, vmin=0, vmax=1, cmap="Blues")
ax2.set_title("Enlarged: %s (highest macro-F1 %s)" % (best, summary[best]["macro_f1"]))
ax2.set_xticks(range(len(cols))); ax2.set_xticklabels(cols)
ax2.set_yticks(range(len(cols))); ax2.set_yticklabels(cols)
ax2.set_xlabel("prediction"); ax2.set_ylabel("reference")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, "%.2f" % cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > 0.5 else "black")
plt.tight_layout()
plt.savefig("/content/confusion_matrix_best.png", dpi=150)
plt.show()'''))

# ----------------------------------------------------------------- mcnemar
CELLS.append(md('''## 15 · Exact pairwise McNemar tests

For every pair of approaches we build the 2×2 table *both correct / A only /
B only / both wrong* over the same 300 reviews, then run the **exact**
McNemar test (binomial) with a statistics package. A small p-value means the
difference in correct/incorrect predictions is unlikely to be chance.'''))
CELLS.append(code('''from statsmodels.stats.contingency_tables import mcnemar
import itertools

results = []
for (na, da), (nb, db) in itertools.combinations(frames.items(), 2):
    m = pd.merge(da[["filtered_index", "reference_label", "prediction"]],
                 db[["filtered_index", "prediction"]], on="filtered_index",
                 suffixes=("_a", "_b"))
    a_ok = (m["prediction_a"] == m["reference_label"]).astype(int)
    b_ok = (m["prediction_b"] == m["reference_label"]).astype(int)
    both_correct = int(((a_ok == 1) & (b_ok == 1)).sum())
    only_a = int(((a_ok == 1) & (b_ok == 0)).sum())
    only_b = int(((a_ok == 0) & (b_ok == 1)).sum())
    neither = int(((a_ok == 0) & (b_ok == 0)).sum())
    table = [[both_correct, only_a], [only_b, neither]]
    res = mcnemar(table, exact=True)
    results.append((na + " vs " + nb, both_correct, only_a, only_b, neither, res.pvalue))
    print("%-12s vs %-12s: both=%3d %-4s-only=%3d %-4s-only=%3d neither=%3d  exact-McNemar p=%.4f"
          % (na, nb, both_correct, na[:4], only_a, nb[:4], only_b, neither, res.pvalue))
print()
print("Interpretation: p < 0.05 -> the difference in correct/incorrect pairs is "
      "statistically significant; larger p -> the two approaches behave alike "
      "on these 300 reviews.")'''))

# ----------------------------------------------------------------- invalid audit
CELLS.append(md('''## 16 · Invalid-response audit

Every invalid response is counted and inspected (nothing is discarded or
guessed). The assignment only requires an INVALID column if any occur.'''))
CELLS.append(code('''for name, df in frames.items():
    inv = df[df["prediction"] == "INVALID"]
    print("=== %s: %d invalid ===" % (name, len(inv)))
    for _, r in inv.head(5).iterrows():
        print("  fi=%d ref=%s raw=%r" % (r["filtered_index"], r["reference_label"], str(r["raw_response"])[:80]))'''))

# ----------------------------------------------------------------- report
CELLS.append(md('''## 17 · Report (nontechnical reader)

The numbers below are pulled directly from the computed outputs above, so the
report always matches the final saved results.'''))
CELLS.append(code('''from IPython.display import Markdown, display
best = max(summary, key=lambda k: summary[k]["macro_f1"])
report = (
"## Report — Comparing zero-shot, few-shot, and fine-tuning\\n\\n"
"**What we did.** We asked three systems to label 300 Amazon gift-card reviews "
"(100 positive, 100 neutral, 100 negative by star rating): the hosted 35B "
"model with no examples (zero-shot), the same model with six labeled examples "
"(few-shot), and a small 3-billion-parameter Llama model that we fine-tuned "
"ourselves on 900 training reviews. All three were scored on the exact same "
"reviews, with the same instructions, greedy decoding, and thinking disabled.\\n\\n"
"**Models and compute.** The hosted model is `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` "
"(35B total, about 3B active). The fine-tuned model is Llama-3.2-3B-Instruct, "
"trained with QLoRA 4-bit adapters (r=16) for 3 epochs on a T4 16 GB GPU, then "
"reloaded from the submitted adapter ZIP for scoring.\\n\\n"
"**Results.**\\n\\n"
"| Approach | Balanced accuracy | Macro-F1 | Correct / 300 |\\n"
"|---|---|---|---|\\n"
"| Zero-shot | %s | %s | %d |\\n"
"| Few-shot | %s | %s | %d |\\n"
"| Fine-tuned | %s | %s | %d |\\n\\n"
"**Which approach won and why.** %s had the highest macro-F1 (%s). "
"[FILL: 2-4 sentences with specific example errors — quote rows from the "
"confusion matrices and invalid audits above.]\\n\\n"
"**What the statistics support.** Pairwise exact McNemar p-values are printed "
"above; [FILL: say which pairs differ significantly and what that does/doesn't "
"prove — 300 reviews is a limited sample and the hosted model answers vary "
"between runs].\\n\\n"
"**Was fine-tuning worth it?** [FILL: compare the fine-tuned score against the "
"hosted approaches and the T4 training time; the adapter ZIP is a few MB and "
"runs locally without any API call.]\\n\\n"
"**How we chose and what we checked.** We chose Llama-3.2-3B for its documented "
"Unsloth support and T4 fit, kept the star rating out of every model input, "
"verified the fixed 900/300/300 splits (unique IDs, no overlap) before "
"training, drew the six few-shot examples only from the training set, and "
"recorded every invalid response instead of guessing. One issue investigated "
"with the agent: the hosted model is a reasoning model that silently spends a "
"small token budget on its thinking chain and returns an empty answer — we "
"verified the `enable_thinking:false` setting against a real probe call before "
"building the runs, and the evaluation uses it everywhere.\\n\\n"
"*Hosted-model results may vary between runs; identical rerun scores are not "
"required (and were not forced).*"
) % (summary["zero-shot"]["balanced_acc"], summary["zero-shot"]["macro_f1"], summary["zero-shot"]["correct"],
     summary["few-shot"]["balanced_acc"], summary["few-shot"]["macro_f1"], summary["few-shot"]["correct"],
     summary["fine-tuned"]["balanced_acc"], summary["fine-tuned"]["macro_f1"], summary["fine-tuned"]["correct"],
     best, summary[best]["macro_f1"])
display(Markdown(report))'''))

# ----------------------------------------------------------------- deliverables
CELLS.append(md('''## 18 · Submission files & final agreement check

The four deliverables, plus a check that they agree:
- This executed notebook (`.ipynb`) — download from **File → Download**
- `assignment4_adapter.zip` — only the two adapter files
- `predictions_fine_tuned.csv` — the 300 fine-tuned test predictions
- `prompt_examples.jsonl` — three real prompts, one per reference class
'''))
CELLS.append(code('''import json
ft = pd.read_csv("/content/predictions_fine_tuned.csv")
assert len(ft) == 300 and ft["prediction"].isin(["POSITIVE", "NEUTRAL", "NEGATIVE", "INVALID"]).all()
pe = [json.loads(l) for l in open("/content/prompt_examples.jsonl", encoding="utf-8")]
assert len(pe) == 3 and {p["reference_label"] for p in pe} == {"POSITIVE", "NEUTRAL", "NEGATIVE"}
assert all(p["filtered_index"] in set(ft["filtered_index"]) for p in pe)

recheck = balanced_accuracy_score(ft["reference_label"], ft["prediction"])
rmac = f1_score(ft["reference_label"], ft["prediction"], average="macro",
                labels=CLASSES, zero_division=0)
assert abs(recheck - summary["fine-tuned"]["balanced_acc"]) < 1e-9
assert abs(rmac - summary["fine-tuned"]["macro_f1"]) < 1e-9
print("AGREEMENT CHECKS PASSED: predictions file reproduces the reported fine-tuned scores;")
print("prompt examples are real rows with one entry per reference class.")

from google.colab import files
for path in ("/content/assignment4_adapter.zip", "/content/predictions_fine_tuned.csv",
             "/content/prompt_examples.jsonl"):
    print("download:", path)
    files.download(path)
print("Also: File -> Download (.ipynb) for the notebook itself. Then keep all four together.")
print("Everything durable is also backed up in your Drive checkpoint folder:", CKPT_DIR)'''))

# ----------------------------------------------------------------- write + validate
nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "Assignment_4_FineTuning_Comparison.ipynb", "provenance": []},
    },
    "cells": CELLS,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT, "| cells:", len(CELLS), "| bytes:", OUT.stat().st_size)

for i, c in enumerate(CELLS):
    if c["cell_type"] == "code":
        # Jupyter magics (lines starting with ! or %) are not Python syntax;
        # strip them before the local syntax check.
        src_lines = []
        for line in "".join(c["source"]).splitlines():
            s = line.strip()
            # IPython magics start with ! or %id/%%id; a line like
            # "% (row[..." is a Python modulo continuation, NOT a magic.
            is_magic = s.startswith("!") or (
                s.startswith("%") and len(s) > 1
                and not s[1].isspace() and s[1] != "("
            )
            if is_magic:
                src_lines.append("# magic: " + s[:40])
            else:
                src_lines.append(line)
        try:
            compile("\n".join(src_lines), "<cell %d>" % i, "exec")
        except SyntaxError as e:
            print("SYNTAX ERROR in cell", i, "line", e.lineno, "col", e.offset)
            print("----- source repr (first 400 chars) -----")
            print(repr("\n".join(src_lines)[:400]))
            print("----- source repr (last 400 chars) -----")
            print(repr("\n".join(src_lines)[-400:]))
            raise
print("all code cells compile clean")
