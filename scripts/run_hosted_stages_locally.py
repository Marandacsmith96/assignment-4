#!/usr/bin/env python3
"""Run the notebook's ZERO-SHOT and FEW-SHOT stages locally (no GPU needed).

These stages only call the hosted class API, so they can run on any machine
with internet. The runner executes the NOTEBOOK'S OWN CELL CODE verbatim
(data prep, checks, prompt config, helpers, run_approach) with paths
remapped from /content to a local staging dir. Outputs are the exact
predictions_*.csv files the notebook's Drive checkpoint folder expects, so
when the notebook runs later it prints "REUSED saved Drive results" and
skips straight to fine-tuning.

Methodology is identical to the notebook: same fixed splits, same prompts,
same generation settings (temp 0, thinking off via extra_body), INVALID
recorded never guessed, responses cached to disk.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NB = json.loads((REPO / "assignment4_finetune_comparison.ipynb").read_text(encoding="utf-8"))
CODE = [c for c in NB["cells"] if c["cell_type"] == "code"]
OUT_DIR = REPO / "data" / "local_stage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GZ = REPO / "data" / "Gift_Cards.jsonl.gz"
assert GZ.exists(), "local dataset missing: %s" % GZ


def cell_src(i: int) -> str:
    return "".join(CODE[i]["source"])


def shim(src: str) -> str:
    # Google Drive mount cell -> local checkpoint dir via A4_CKPT_DIR
    src = src.replace("from google.colab import drive\n",
                      "pass  # no google.colab locally\n")
    src = src.replace('if not os.path.ismount("/content/drive"):\n    drive.mount("/content/drive")\n',
                      "pass  # no Drive mount locally\n")
    # dataset location + no re-download locally
    src = src.replace('GZ = Path("/content/Gift_Cards.jsonl.gz")',
                      'GZ = Path(r"%s")' % GZ)
    src = src.replace("urllib.request.urlretrieve(DATA_URL, GZ)",
                      "pass  # local dataset already present")
    # all /content paths -> local staging dir
    src = src.replace('"/content/', '"%s/' % OUT_DIR.as_posix().replace("\\", "/"))
    src = src.replace("' /content/", "' %s/" % OUT_DIR.as_posix().replace("\\", "/"))
    return src


# key from .env (never printed)
key = None
for line in (REPO.parent / "Assignment-2-Group-I" / ".env").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\s*MBAX_API_KEY\s*=\s*[\"']?([^\"'\s]+)", line)
    if m:
        key = m.group(1).strip()
        break
assert key, "MBAX_API_KEY not found in Assignment-2-Group-I/.env"
os.environ["DOBOLYI_CLASS_KEY"] = key
os.environ["A4_CKPT_DIR"] = str(OUT_DIR / "ckpts")

# assemble the notebook flow (code-cell indices; skipping runtime/install cell)
parts = []
for idx, label in [(1, "credentials"), (2, "drive/checkpoint helpers"),
                   (3, "data prep"), (4, "data checks"), (5, "formatted examples"),
                   (6, "prompt config"), (7, "scoring helpers"),
                   (8, "zero-shot"), (9, "few-shot")]:
    parts.append("\n# ===== notebook cell %d (%s) =====\n" % (idx, label))
    parts.append(shim(cell_src(idx)))

ns = {"__name__": "__main__"}
src_all = "\n".join(parts)
try:
    exec(compile(src_all, "local_hosted_stages.py", "exec"), ns)
except SystemExit:
    raise
finally:
    pass

# ---- report what was produced (traceability) ----
import csv as _csv
ck = OUT_DIR / "ckpts"
for name in ("zero_shot", "few_shot"):
    p = ck / ("predictions_%s.csv" % name)
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        inv = sum(r["prediction"] == "INVALID" for r in rows)
        print("\nRESULT %s: %d rows, %d invalid -> %s" % (name, len(rows), inv, p))
    else:
        print("\nRESULT %s: MISSING (%s)" % (name, p))
print("\nUpload BOTH predictions_*.csv from %s into Drive/assignment4_ckpts" % ck)
