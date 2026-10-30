#!/usr/bin/env python3
"""Prepare and VERIFY the fixed Assignment-4 splits for Amazon Reviews '23 Gift_Cards.

Implements the assignment rules EXACTLY (MBAX 6418 Assignment 4):
- Labels: 4-5 stars = POSITIVE; 3 = NEUTRAL; 1-2 = NEGATIVE.
- Preserve file order; original line number as original_row, starting at 1.
- Skip blank lines and reviews with fewer than three characters of text
  after trimming whitespace (text = trimmed title + ' ' + trimmed text,
  matching Assignment 1's title+text input convention); missing text = "".
- Number the remaining reviews as filtered_index, starting at 0.
- Train  = filtered_index 1000..19999: sample 300/class via
           random.Random(43).sample() on each class's file-ordered pool,
           in class order POSITIVE, NEUTRAL, NEGATIVE, then shuffle the
           combined 900 with that same generator.
- Val    = filtered_index 20000..29999: first 100 per class (reserved,
           NOT used in this assignment).
- Test   = filtered_index 30000..39999: first 100 per class (used by all
           three approaches).
Runs on stdlib only (gzip/json/random/csv) so it works locally and in Colab.

Usage:  python prepare_splits.py [path/to/Gift_Cards.jsonl.gz]
Output: prints a verification report; writes data/train_ids.csv,
        data/val_ids.csv, data/test_ids.csv (filtered_index + original_row
        + label) for manual spot-checking.
"""
import csv
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GZ = HERE.parent / "data" / "Gift_Cards.jsonl.gz"

CLASS_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def label_of(rating: int) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def load_filtered(gz_path: Path):
    """Return (rows, skipped) with the assignment's filtering rules."""
    rows = []
    skipped = Counter()
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:                      # blank line -> skip
                skipped["blank"] += 1
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:   # unparseable line -> skip
                skipped["json"] += 1
                continue
            rating = obj.get("rating")
            try:
                rating = int(round(float(rating)))
            except (TypeError, ValueError):
                skipped["no_rating"] += 1
                continue
            title = obj.get("title")
            text = obj.get("text")
            title = title if isinstance(title, str) else ""
            text = text if isinstance(text, str) else ""
            content = f"{title} {text}".strip()      # missing text = ""
            if len(content) < 3:                     # <3 chars of text -> skip
                skipped["short"] += 1
                continue
            rows.append(
                {
                    "original_row": lineno,
                    "rating": rating,
                    "title": title,
                    "text": text,
                    "content": content,
                    "label": label_of(rating),
                }
            )
    return rows, skipped


def first_per_class(pool, k=100, order=CLASS_ORDER):
    """First k rows of each class in a file-ordered block (val/test rule)."""
    out = []
    used = set()
    for cls in order:
        n = 0
        for r in pool:
            if r["label"] == cls and n < k:
                out.append(r)
                used.add(r["id"])
                n += 1
    return out


def main(gz_path: Path):
    rows, skipped = load_filtered(gz_path)
    for i, r in enumerate(rows):
        r["id"] = i                      # filtered_index, starting at 0
    N = len(rows)

    train_block = rows[1000:20000]       # filtered_index 1000..19999
    val_block = rows[20000:30000]        # 20000..29999
    test_block = rows[30000:40000]       # 30000..39999

    rng = random.Random(43)              # THE fixed generator
    train = []
    for cls in CLASS_ORDER:              # sample per class, this order
        pool = [r for r in train_block if r["label"] == cls]
        train.extend(rng.sample(pool, 300))
    rng.shuffle(train)                   # same generator shuffles the 900

    val = first_per_class(val_block)
    test = first_per_class(test_block)

    # ---- verification -------------------------------------------------
    def ids(recs):
        return [r["id"] for r in recs]

    t_ids, v_ids, e_ids = ids(train), ids(val), ids(test)
    t_set, v_set, e_set = set(t_ids), set(v_ids), set(e_ids)

    print(f"file: {gz_path.name}")
    print(f"filtered rows (after rules): {N:,}   skipped: {dict(skipped)}")
    print(f"train block rows (1000..19999): {len(train_block):,}")
    print(f"val   block rows (20000..29999): {len(val_block):,}")
    print(f"test  block rows (30000..39999): {len(test_block):,}")
    print()
    print(f"TRAIN drawn: {len(train)}   classes: {Counter(r['label'] for r in train)}")
    print(f"VAL  drawn: {len(val)}   classes: {Counter(r['label'] for r in val)}")
    print(f"TEST drawn: {len(test)}   classes: {Counter(r['label'] for r in test)}")
    print()
    assert len(train) == 900, "train != 900"
    assert Counter(r["label"] for r in train) == Counter(
        {"POSITIVE": 300, "NEUTRAL": 300, "NEGATIVE": 300}
    ), "train not 300/class"
    assert len(val) == 300 and len(test) == 300, "val/test != 300"
    assert all(Counter(r["label"] for r in s) == Counter(
        {"POSITIVE": 100, "NEUTRAL": 100, "NEGATIVE": 100}
    ) for s in (val, test)), "val/test not 100/class"
    assert len(t_set) == 900 and len(v_set) == 300 and len(e_set) == 300, "dup ids"
    assert not (t_set & v_set) and not (t_set & e_set) and not (v_set & e_set), "overlap!"
    print("CHECKS PASSED: 900/300/300, 300+300+300 per class, unique IDs, "
          "no overlap between sets.")
    print()
    print("few-shot candidates (first 2 PER CLASS in the FINAL shuffled train):")
    printed = {c: 0 for c in CLASS_ORDER}
    for r in train:
        if printed[r["label"]] < 2:
            print(f"  [{r['label']:8s}] filtered_index={r['id']:>6d} "
                  f"orig_row={r['original_row']:>7d} rating={r['rating']} "
                  f"title={r['title'][:40]!r}")
            printed[r["label"]] += 1
        if all(v == 2 for v in printed.values()):
            break

    out_dir = HERE.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for name, recs in (("train_ids.csv", train), ("val_ids.csv", val),
                       ("test_ids.csv", test)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filtered_index", "original_row", "label", "rating"])
            for r in recs:
                w.writerow([r["id"], r["original_row"], r["label"], r["rating"]])
        print(f"wrote data/{name} ({len(recs)} rows)")
    return train, val, test


if __name__ == "__main__":
    gz = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GZ
    main(gz)
