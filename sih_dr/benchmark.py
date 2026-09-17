"""M6 benchmark: CNN-only vs features-only vs hybrid on the SEALED test split.

Honest-by-construction: arms without their required artifact report
'pending' with the reason instead of a number. Never trains on TEST.

- features-only: LogReg on M2 lesion_features (fit on TRAIN, scored on TEST)
- cnn-only:     torch checkpoint in models/ (probs on TEST; skipped if absent)
- hybrid:       LogReg stack of [cnn_probs..., feat_prob, heuristic_score]
                (needs cnn; skipped if absent)

Usage:  python -m sih_dr.benchmark [--max-train N] [--max-test N]
Writes: results/benchmark.json + results/benchmark.md (paste into PPT)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FEAT_ORDER = ["ma_count", "ma_area", "ex_count", "ex_area",
              "he_count", "he_area", "vessel_density", "nv_present"]


def feat_vec(lf):
    return [float(lf[k]) for k in FEAT_ORDER]


def featurize(files, imgdir, limit, want_cnn=False):
    from .run_pipeline import screen_image
    cnn_ok = False
    if want_cnn:
        from .m3_grade import cnn_probs_only
        probe, _ = cnn_probs_only(np.zeros((8, 8, 3), np.uint8))
        cnn_ok = probe is not None
        if not cnn_ok:
            print("  cnn inference unavailable — arm will report pending")
    X, rows, C = [], [], []
    for i, fn in enumerate(files[:limit]):
        img = cv2.imread(str(imgdir / fn))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        try:
            r = screen_image(img)
        except Exception as e:
            print(f"  skip {fn}: {e}", file=sys.stderr)
            continue
        lf = r["seg"]["lesion_features"]
        g = r["grading"]
        X.append(feat_vec(lf))
        rows.append({"file": fn, "sec": round(time.time() - t0, 2),
                     "heuristic_probs": g["class_probabilities"]})
        if cnn_ok:
            cp, _ = cnn_probs_only(r["m1"]["enhanced_image"])
            C.append(cp)
    return np.array(X), rows, (np.array(C) if cnn_ok else None)


def metrics_referable(y_true, y_score):
    from .run_pipeline import referable_metrics
    return referable_metrics(list(y_true), list(y_score))


def main():
    from .splits import DATA, load_splits
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=10 ** 9)
    ap.add_argument("--max-test", type=int, default=10 ** 9)
    a = ap.parse_args()
    sp = load_splits()
    imgdir = DATA / "aptos2019" / "train_images"
    arms = {}

    for split in ("train", "val", "test"):
        files = sp["splits"][split]
        lim = {"train": a.max_train, "val": a.max_train,
               "test": a.max_test}[split]
        print(f"featurizing {split} ({min(len(files), lim)} imgs)...")
        X, rows, C = featurize(files, imgdir, lim, want_cnn=True)
        y = np.array([sp["labels"][r["file"]] for r in rows])
        arms[split] = (X, y, rows, C)
    Xtr, ytr, _, Ctr = arms["train"]
    Xva, yva, _, Cva = arms["val"]
    Xte, yte, rows_te, Cte = arms["test"]

    from sklearn.linear_model import LogisticRegression
    out = {"arms": {}, "n_train": len(ytr), "n_test": len(yte)}

    # --- features-only arm ---
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, (ytr >= 2).astype(int))
    feat_score = clf.predict_proba(Xte)[:, 1]
    out["arms"]["features-only"] = {
        "status": "ok", **metrics_referable(yte, feat_score),
        "note": "LogReg on 8 M2 lesion features, fit on TRAIN"}

    # --- cnn-only arm (real inference; pending without checkpoint) ---
    from .m3_grade import cnn_probs_only
    _, reason = cnn_probs_only(np.zeros((8, 8, 3), np.uint8))
    cnn_ok = Cte is not None
    if cnn_ok:
        out["arms"]["cnn-only"] = {
            "status": "ok", **metrics_referable(yte, Cte[:, 2:].sum(1)),
            "note": "EfficientNet checkpoint, isolated image branch"}
    else:
        out["arms"]["cnn-only"] = {
            "status": "pending", "note": reason or "cnn unavailable"}

    # --- hybrid arm: LogReg stack fit on VAL (never TEST) ---
    if cnn_ok:
        feat_val = clf.predict_proba(Xva)[:, 1]
        stack_va = np.column_stack([Cva, feat_val])
        stacker = LogisticRegression(max_iter=2000)
        stacker.fit(stack_va, (yva >= 2).astype(int))
        feat_te = feat_score
        stack_te = np.column_stack([Cte, feat_te])
        hybrid_score = stacker.predict_proba(stack_te)[:, 1]
        out["arms"]["hybrid"] = {
            "status": "ok", **metrics_referable(yte, hybrid_score),
            "note": "LogReg stack of cnn probs + lesion-feature prob, fit on VAL"}
    else:
        out["arms"]["hybrid"] = {
            "status": "pending",
            "note": "needs cnn-only arm first — integrated claim unproven until then"}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps(out, indent=1))
    lines = ["# Benchmark — referable DR (grade ≥ 2), sealed APTOS test split",
             f"n_train={out['n_train']} n_test={out['n_test']}", "",
             "| arm | status | sens | spec | auc | note |",
             "|---|---|---|---|---|---|"]
    for name, r in out["arms"].items():
        if r["status"] == "ok":
            lines.append(f"| {name} | ok | {r['sens']} | {r['spec']} | {r['auc']} | {r['note']} |")
        else:
            lines.append(f"| {name} | pending | — | — | — | {r['note']} |")
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
