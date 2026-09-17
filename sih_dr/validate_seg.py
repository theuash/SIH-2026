"""Segmentation validation vs DRIVE (vessels) + IDRiD-B (MA/HE/EX/OD).

Usage:  python -m sih_dr.validate_seg [--max N] [--scale 1024]
Writes: results/seg_metrics.json + results/seg_metrics.md

Conventions (case-insensitive globs, any of .tif/.gif/.png/.jpg):
  data/drive/{training,test}/images/*, */1st_manual/*, */mask/*
  data/idrid/segmentation/{train,test}/images/* + *_{MA,HE,EX,OD}.* masks
Metrics: vessels Dice/sens/spec; lesions Dice + GT-component recall;
OD centroid error (px and % width). Downscales long side to --scale for speed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 127) if img.dtype == bool else (img > 0)


def find(d, *names):
    if not d or not d.exists():
        return []
    if not names:
        names = ("",)
    out = []
    for n in names:
        out += sorted(d.rglob(f"*{n}.*"))
    seen, uniq = set(), []
    for p in out:
        if p.suffix.lower() in {".tif", ".tiff", ".gif", ".png", ".jpg", ".jpeg"} \
                and p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq


def prep(img, scale):
    h, w = img.shape[:2]
    s = min(1.0, scale / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, s


def dice(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if not a.any() and not b.any():
        return 1.0
    return round(2 * (a & b).sum() / max(a.sum() + b.sum(), 1), 4)


def comp_recall(pred, gt):
    """Fraction of GT components touched by any predicted pixel."""
    n, lab, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    if n <= 1:
        return None
    hit = sum(bool((pred[lab == i]).any()) for i in range(1, n))
    return round(hit / (n - 1), 4)


def main():
    from .splits import DATA
    from .run_pipeline import screen_image
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10 ** 9)
    ap.add_argument("--scale", type=int, default=1024)
    a = ap.parse_args()
    out = {"vessels": [], "lesions": {}, "od_err": []}

    # --- DRIVE vessels ---
    for split in ("training", "test"):
        d = DATA / "drive" / split
        imgs = find(d / "images")
        man = {p.name.split("_")[0]: p for p in find(d / "1st_manual")}
        for ip in imgs[:a.max]:
            key = ip.name.split("_")[0]
            if key not in man:
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            img, s = prep(img, a.scale)
            gt = load_gray(man[key])
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            try:
                seg = screen_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["seg"]
            except Exception as e:
                print(f"  skip {ip.name}: {e}", file=sys.stderr)
                continue
            pr = seg["vessel_mask"]
            tp = float((pr & gt).sum())
            out["vessels"].append({
                "file": ip.name, "dice": dice(pr, gt),
                "sens": round(tp / max(gt.sum(), 1), 4),
                "spec": round(((~pr) & (~gt)).sum() / max((~gt).sum(), 1), 4)})

    # --- IDRiD-B lesions + OD ---
    for split in ("train", "test"):
        d = DATA / "idrid" / "segmentation" / split
        for ip in find(d / "images")[:a.max]:
            stem = ip.stem
            img = cv2.imread(str(ip))
            if img is None:
                continue
            img, s = prep(img, a.scale)
            try:
                seg = screen_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["seg"]
            except Exception as e:
                print(f"  skip {ip.name}: {e}", file=sys.stderr)
                continue
            pred = {"MA": seg["microaneurysm_map"]["mask"],
                    "HE": seg["hemorrhage_mask"], "EX": seg["exudate_mask"]}
            for tag, pm in pred.items():
                hits = find(d, f"{stem}_{tag}", f"{stem}*{tag}")
                if not hits:
                    continue
                gt = load_gray(hits[0])
                gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                out["lesions"].setdefault(tag, []).append({
                    "file": ip.name, "dice": dice(pm, gt),
                    "recall": comp_recall(pm, gt)})
            od_hits = find(d, f"{stem}_OD", f"{stem}*OD")
            if od_hits:
                gt = load_gray(od_hits[0])
                gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                ys, xs = np.nonzero(gt)
                if len(xs):
                    cx, cy = seg["optic_disc"]["centroid"]
                    err = ((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) ** 0.5
                    out["od_err"].append({"file": ip.name,
                                          "px": round(float(err), 1),
                                          "pct_width": round(float(err) / img.shape[1] * 100, 2)})

    def mean(key, field="dice"):
        vals = [r[field] for r in key if r[field] is not None]
        if not vals:
            return "—", 0
        return round(sum(vals) / len(vals), 4), len(vals)

    summary = {"vessels_dice": mean(out["vessels"])}
    for tag, rows in out["lesions"].items():
        summary[f"{tag}_dice"] = mean(rows)
        summary[f"{tag}_recall"] = mean(rows, "recall")
    if out["od_err"]:
        errs = [r["pct_width"] for r in out["od_err"]]
        summary["od_err_pct_width"] = (round(sum(errs) / len(errs), 2), len(errs))
    final = {"summary": summary, "detail": out}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "seg_metrics.json").write_text(json.dumps(final, indent=1))
    md = ["# Segmentation validation", "",
          "| target | metric | value | n |",
          "|---|---|---|---|"]
    for k, (v, n) in summary.items():
        md.append(f"| {k} | mean | {v} | {n} |")
    (RESULTS / "seg_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
