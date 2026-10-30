#!/usr/bin/env python3
"""Local smoke test for the Assignment-4 notebook's core logic.

Runs the notebook's OWN data-prep + checks cells (path-swapped to the local
dataset) and, if pandas/sklearn/statsmodels are importable here, the
evaluation cells against clearly-marked SYNTHETIC predictions. This finds
code bugs before the notebook runs in Colab. Nothing here produces submission
numbers — it is a code check only.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NB = json.loads((REPO / "assignment4_finetune_comparison.ipynb").read_text(encoding="utf-8"))

cells = [c for c in NB["cells"] if c["cell_type"] == "code"]


def cell_src(index: int) -> str:
    return "".join(cells[index]["source"])


def run(name: str, src: str) -> None:
    print(f"\n===== {name} =====")
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=600)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
        raise SystemExit(f"{name} FAILED (exit {r.returncode})")
    print(f"----- {name} OK -----")


# --- notebook code-cell order (indices within 'code' cells) ---
# 0 runtime, 1 credentials, 2 drive/checkpoints, 3 data load+split, 4 checks,
# 5 fmt, 6 prompt cfg, 7 helpers, 8 zero-shot, 9 few-shot, 10 ft load,
# 11 train, 12 zip, 13 reload, 14 ft predictions, 15 metrics, 16 confusion,
# 17 mcnemar, 18 invalid audit, 19 report, 20 deliverables

# --- 0) checkpointer round-trip (Drive-cell helpers, local paths) ---
import os
smoke_dir = REPO / "data" / "smoke"
smoke_dir.mkdir(exist_ok=True)
ckpt_src = cell_src(2)
ckpt_src = ckpt_src.replace("from google.colab import drive\n", "pass  # google.colab not available locally\n")
ckpt_src = ckpt_src.replace('if not os.path.ismount("/content/drive"):\n    drive.mount("/content/drive")\n',
                            "pass  # no Drive mount in the local smoke run\n")
os.environ["A4_CKPT_DIR"] = str(smoke_dir / "ckpts")
roundtrip = ckpt_src + '''
# --- round trip: part-file append -> renamed to finished -> stage_complete ---
import shutil
shutil.rmtree(CKPT_DIR, ignore_errors=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
recs = [{"filtered_index": i, "prediction": p, "raw_response": "x"}
        for i, p in enumerate(["POSITIVE", "NEUTRAL", "NEGATIVE"])]
for r in recs:
    _append_part("zt", r)
assert stage_part("zt").exists() and not stage_complete("zt", expected=3), "part file must NOT count as finished"
shutil.copyfile(stage_part("zt"), stage_csv("zt"))
assert stage_complete("zt", expected=3), "finished stage must be detected"
assert not stage_complete("zt", expected=4), "wrong row count must mean re-run"
print("CHECKPOINT ROUND-TRIP OK")
'''
run("CHECKPOINTER ROUND-TRIP (Drive cell helpers, local paths)", roundtrip)

# --- 1) data-prep cell: swap /content -> local repo data, strip google bits
data_src = cell_src(3)
data_src = data_src.replace('GZ = Path("/content/Gift_Cards.jsonl.gz")',
                            'GZ = Path(r"%s")' % (REPO / "data" / "Gift_Cards.jsonl.gz"))
data_src = data_src.replace('urllib.request.urlretrieve(DATA_URL, GZ)',
                            'pass  # local copy already present')
run("DATA PREP (notebook cell, local path)", data_src)

# --- 2) checks cell: depends on names from data cell -> run both together
#     redirect /content writes to a local smoke dir
checks_src = cell_src(4).replace("/content/", smoke_dir.as_posix() + "/")
run("DATA CHECKS (notebook cell)", data_src + "\n\n" + checks_src)

# --- 3) prompt-config cell (needs train/t_ids from above too)
cfg_src = cell_src(6)
run("PROMPT CONFIG + FEW-SHOT VERIFICATION", data_src + "\n\n" + checks_src + "\n\n" + cfg_src)

# --- 4) evaluation cells with synthetic predictions (marked as smoke test)
try:
    import pandas  # noqa
    import sklearn  # noqa
    import statsmodels  # noqa
    HAVE_ML = True
except Exception:
    HAVE_ML = False
    print("\n(sklearn/statsmodels not available locally — eval smoke test skipped)")

if HAVE_ML:
    synth = '''
import pandas as pd, numpy as np
import collections
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
rng2 = np.random.RandomState(7)
# synthetic: 300 rows, order of the prescribed test ids (ids not required for the code path check)
test_rows = [{"filtered_index": i, "label": CLASSES[i % 3]} for i in range(300)]
def synth_df(acc):
    df = []
    for i, r in enumerate(test_rows):
        pred = r["label"] if rng2.rand() < acc else CLASSES[int(rng2.randint(0, 3))]
        df.append({"filtered_index": i, "reference_label": r["label"], "prediction": pred,
                   "raw_response": pred})
    return pd.DataFrame(df)
frames = {"zero-shot": synth_df(0.55), "few-shot": synth_df(0.60), "fine-tuned": synth_df(0.68)}
e_ids = {i for i in range(300)}
'''
    eval_src = synth + "\n\n" + cell_src(15) + "\n\n" + cell_src(16) + "\n\n" + cell_src(18)
    # neutralize the prediction-file LOADS (frames come from the harness)
    load_block = ('frames = {"zero-shot": load_pred("/content/predictions_zero_shot.csv"),\n'
                  '          "few-shot": load_pred("/content/predictions_few_shot.csv"),\n'
                  '          "fine-tuned": load_pred("/content/predictions_fine_tuned.csv")}')
    eval_src = eval_src.replace(load_block, "# frames injected by smoke harness")
    eval_src = eval_src.replace('"/content/metrics_summary.json"', '"%s"' % (smoke_dir.as_posix() + "/metrics_summary.json"))
    eval_src = eval_src.replace('"/content/confusion_matrices_3x.png"', '"%s"' % (smoke_dir.as_posix() + "/confusion_matrices_3x.png"))
    eval_src = eval_src.replace('"/content/confusion_matrix_best.png"', '"%s"' % (smoke_dir.as_posix() + "/confusion_matrix_best.png"))
    run("EVAL CELLS (metrics/confusion/mcnemar) with SYNTHETIC predictions", eval_src)

print("\nALL LOCAL SMOKE TESTS PASSED")
