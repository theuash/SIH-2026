"""Fixed train/val/test splits. Run once when data lands; results committed.

- APTOS: stratified 80/10/10 by grade (seed 26038). TEST is sealed: benchmark.py
  is the only consumer allowed to read it, never training.
- IDRiD segmentation + DRIVE: used whole for seg validation (too small to split).
- Messidor-2: cross-dataset generalization check (train APTOS, test Messidor-2).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 26038


def aptos_labels():
    csvp = DATA / "aptos2019" / "train.csv"
    if not csvp.exists():
        return {}
    with open(csvp) as f:
        return {r["id_code"]: int(r["diagnosis"]) for r in csv.DictReader(f)}


def make_splits():
    import random
    labels = aptos_labels()
    by_grade = defaultdict(list)
    imgdir = DATA / "aptos2019" / "train_images"
    for code, g in labels.items():
        for ext in (".png", ".jpg", ".jpeg"):
            p = imgdir / (code + ext)
            if p.exists():
                by_grade[g].append(p.name)
                break
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    labelmap = {}
    for g, files in sorted(by_grade.items()):
        rng.shuffle(files)
        n = len(files)
        n_test, n_val = max(1, n // 10), max(1, n // 10)
        splits["test"] += files[:n_test]
        splits["val"] += files[n_test:n_test + n_val]
        splits["train"] += files[n_test + n_val:]
        for fn in files:
            labelmap[fn] = g
    out = {"seed": SEED, "splits": splits, "labels": labelmap,
           "note": "TEST sealed: only benchmark.py may read it"}
    (DATA / "splits.json").write_text(json.dumps(out, indent=1))
    print({k: len(v) for k, v in splits.items()}, f"({len(labelmap)} labeled)")
    return out


def load_splits():
    p = DATA / "splits.json"
    if not p.exists():
        raise SystemExit("no data/splits.json — add data, then: python -m sih_dr.splits")
    return json.loads(p.read_text())


if __name__ == "__main__":
    if not aptos_labels():
        raise SystemExit("data/aptos2019/train.csv missing — run scripts/fetch_datasets.sh first")
    make_splits()
