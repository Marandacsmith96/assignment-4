#!/usr/bin/env python3
"""Direct eval-section smoke: synthetic predictions -> metrics/confusion/mcnemar.
Runs the notebook's OWN eval cells (14, 15, 16, 17) against clearly-marked
synthetic data. Verifies the math code paths, not the numbers.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = json.loads((REPO / "assignment4_finetune_comparison.ipynb").read_text(encoding="utf-8"))
code_cells = [c for c in NB["cells"] if c["cell_type"] == "code"]

def src(i):
    return "".join(code_cells[i]["source"])

synth = '''import pandas as pd, numpy as np
import collections
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
rng2 = np.random.RandomState(7)
test_rows = [{"filtered_index": i, "label": CLASSES[i % 3]} for i in range(300)]
e_ids = {i for i in range(300)}
def synth_df(acc):
    df = []
    for i, r in enumerate(test_rows):
        pred = r["label"] if rng2.rand() < acc else CLASSES[int(rng2.randint(0, 3))]
        df.append({"filtered_index": i, "reference_label": r["label"],
                   "prediction": pred, "raw_response": pred})
    return pd.DataFrame(df)
frames = {"zero-shot": synth_df(0.55), "few-shot": synth_df(0.60), "fine-tuned": synth_df(0.68)}
'''

smoke_dir = REPO / "data" / "smoke"
smoke_dir.mkdir(exist_ok=True)

body = ("import matplotlib\nmatplotlib.use(\"Agg\")\n" + synth + "\n\n" + src(14)
        + "\n\n" + src(15) + "\n\n" + src(16) + "\n\n" + src(17))
load_block = ('frames = {"zero-shot": load_pred("/content/predictions_zero_shot.csv"),\n'
              '          "few-shot": load_pred("/content/predictions_few_shot.csv"),\n'
              '          "fine-tuned": load_pred("/content/predictions_fine_tuned.csv")}')
body = body.replace(load_block, "# frames injected by smoke harness")
path_map = {
    '"/content/metrics_summary.json"': smoke_dir.as_posix() + "/metrics_summary.json",
    '"/content/confusion_matrices_3x.png"': smoke_dir.as_posix() + "/confusion_matrices_3x.png",
    '"/content/confusion_matrix_best.png"': smoke_dir.as_posix() + "/confusion_matrix_best.png",
}
for old, new in path_map.items():
    body = body.replace(old, '"%s"' % new)

r = subprocess.run([sys.executable, "-u", "-c", body], capture_output=True, text=True, timeout=600)
print(r.stdout[-4000:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)
print("\nEVAL SMOKE OK")
