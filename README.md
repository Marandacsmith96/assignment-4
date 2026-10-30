# assignment-4

**MBAX 6418 · Assignment 4 — Fine-Tuning: Compare Three Approaches**

Amazon gift-card review classification (POSITIVE / NEUTRAL / NEGATIVE) three
ways on the same 300 test reviews: **zero-shot**, **few-shot**, and
**fine-tuning** (Llama-3.2-3B with Unsloth + QLoRA on a T4 GPU). One notebook,
four submission files.

## Run it (Google Colab, T4)

1. Open `assignment4_finetune_comparison.ipynb` in Colab
   (drag-and-drop the file onto the Colab page works).
2. Add the class API key as a Colab **secret** named `DOBOLYI_CLASS_KEY`
   (left sidebar → key icon → Secrets → notebook access ON).
3. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
   When the Google Drive popup appears, pick your account and click **Allow**
   (one time) — the notebook saves its progress there as it goes.
   ~30–50 minutes end to end. When it finishes, the notebook prints download
   links for the four submission files.
4. **If Colab disconnects mid-run:** don't panic — just **Run all** again.
   Every stage saves to `assignment4_ckpts` in your Drive as it completes,
   so finished stages are skipped and half-done ones resume where they stopped.

Never commit or share the API key — it lives only in the Colab secret.

## The four submission files

| File | Generated where | Contents |
|---|---|---|
| `assignment4_finetune_comparison.ipynb` | this repo | the executed notebook (download from Colab after running) |
| `assignment4_adapter.zip` | Colab | only `adapter_model.safetensors` + `adapter_config.json` |
| `predictions_fine_tuned.csv` | Colab | all 300 test reviews + fine-tuned predictions |
| `prompt_examples.jsonl` | Colab | 3 real prompts, one per reference class |

## What is in the notebook

- Fixed data + splits (Amazon Reviews '23 **Gift Cards**, labels from star
  rating, `random.Random(43)` sampling, 900 train / 300 held-out val / 300
  test) with checks that stop the notebook if anything is off
- Zero-shot and few-shot runs of the hosted class model
  (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`), greedy decoding, thinking disabled
  (the flag is sent via `extra_body` so the current openai SDK transmits it
  reliably — verified against a live probe call), plus a one-call probe that
  prints the first raw answer before the 300-review loop starts, and
  responses cached to disk
- Fine-tuning of `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit QLoRA,
  r=16, 3 epochs) → adapter ZIP → reloaded from the ZIP onto a fresh model
  for scoring (no training, no hosted calls)
- **Disconnect protection:** every expensive stage mirrors its finished
  results to `assignment4_ckpts` in Google Drive as it goes; training
  checkpoints every 50 steps. Re-running the notebook skips finished stages
  and resumes half-done ones — and any saved stage that is ALL INVALID is
  refused and re-scored, so a broken earlier run can never poison results
- Evaluation: balanced accuracy, macro-F1, per-class precision/recall/F1,
  row-normalized confusion matrices (plus an enlarged one for the winner),
  exact pairwise McNemar tests, invalid-response audit
- A nontechnical report with the numbers filled in from the run

## Verification artifacts in this repo

- `data/train_ids.csv`, `data/val_ids.csv`, `data/test_ids.csv` — the exact
  prescribed review IDs (filtered_index, original_row, label, rating) for
  personal spot-checking
- `scripts/prepare_splits.py` — reproduces the split rules and re-checks them
- `scripts/build_notebook.py` — builds the notebook from cell definitions
- `scripts/smoke_test_notebook.py`, `scripts/smoke_eval_only.py` — local
  checks of the notebook's data + evaluation logic (real data / synthetic
  predictions; they catch code bugs, they don't produce submission numbers)

## Decisions (documented)

- **Small model:** Llama-3.2-3B-Instruct — best-documented Unsloth target,
  fits T4 16 GB in 4-bit, strong instruction following for a 3B model.
- **Prompt:** one shared instruction across all three approaches, exactly one
  label requested, temperature 0, 32-token output cap, thinking disabled.
- **Few-shot:** 6 examples, 2 per class, drawn only from the prescribed
  training set (the notebook asserts this).
- **Splits rule detail:** "fewer than three characters of text" is applied to
  the trimmed `title + " " + text` (the actual model input), matching the
  Assignment 1 convention; missing text is treated as empty.
