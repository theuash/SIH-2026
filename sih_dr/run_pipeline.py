"""M6 — run_pipeline.py: folder of images -> results.csv + metrics.

Usage:
  python -m sih_dr.run_pipeline --input data/aptos --out results.csv
  # with labels.csv (columns: file,grade) also prints sens/spec/AUC for referable
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


def screen_image(img):
    from .m1_quality import assessAndEnhance
    from .m2_segment import segmentStructures
    from .m3_grade import gradeDR
    from .m4_explain import explainGrading
    t0 = time.time()
    m1 = assessAndEnhance(img)
    seg = segmentStructures(m1["enhanced_image"])
    g = gradeDR(m1["enhanced_image"], seg["lesion_features"])
    ex = explainGrading(m1["enhanced_image"], g["net"], seg, g)
    dt = round(time.time() - t0, 2)
    return {"m1": m1, "seg": seg, "grading": g, "explain": ex, "sec": dt}


def referable_metrics(y_true, y_score):
    """Sens/spec/AUC for referable DR (grade>=2). sklearn if present, else manual."""
    y = [1 if g >= 2 else 0 for g in y_true]
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, y_score))
    except Exception:
        # Mann-Whitney rank AUC, no deps
        pos = sorted([s for s, t in zip(y_score, y) if t == 1])
        neg = sorted([s for s, t in zip(y_score, y) if t == 0])
        if not pos or not neg:
            return {"sens": 0, "spec": 0, "auc": 0.5}
        wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
            1 for p in pos for n in neg if p == n)
        auc = wins / (len(pos) * len(neg))
    thr = 0.5
    tp = sum(1 for s, t in zip(y_score, y) if s >= thr and t == 1)
    fn = sum(1 for s, t in zip(y_score, y) if s < thr and t == 1)
    tn = sum(1 for s, t in zip(y_score, y) if s < thr and t == 0)
    fp = sum(1 for s, t in zip(y_score, y) if s >= thr and t == 0)
    return {"sens": round(tp / max(tp + fn, 1), 3),
            "spec": round(tn / max(tn + fp, 1), 3), "auc": round(auc, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    files = sorted([p for p in Path(a.input).glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    labels = {}
    if a.labels and Path(a.labels).exists():
        with open(a.labels) as f:
            for row in csv.DictReader(f):
                labels[row["file"]] = int(row["grade"])
    rows, ys, yt = [], [], []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = screen_image(img)
        g = r["grading"]
        rows.append({"file": p.name, "accepted": r["m1"]["accepted"],
                     "quality": r["m1"]["quality_score"], "grade": g["grade"],
                     "referable": g["referable"], "confidence": g["confidence"],
                     "sec": r["sec"], "report": r["explain"]["report_file"]})
        if p.name in labels:
            yt.append(labels[p.name])
            ys.append(sum(g["class_probabilities"][2:]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} images)")
    if yt:
        print("referable metrics:", referable_metrics(yt, ys))


if __name__ == "__main__":
    main()
