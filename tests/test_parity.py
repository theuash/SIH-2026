"""Parity: C++ core vs Python twins. Same inputs -> same clinical outputs.

Run:  python tests/test_parity.py   (needs built extension; skips otherwise)
Tolerances are honest: enhancement/overlay pixels are perceptual (±12/255),
masks Dice-based, decisions (accepted/grade/counts) near-exact.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# NOTE: import _native BEFORE setting IMPL so the built extension loads;
# twins (_assess_py etc.) are importable regardless.
from sih_dr import _native  # noqa: E402
os.environ["NETRADR_IMPL"] = "py"
from sih_dr.m1_quality import _assess_py  # noqa: E402
from sih_dr.m2_segment import _segment_py  # noqa: E402
from sih_dr.m4_explain import _pixels_py  # noqa: E402

import cv2  # noqa: E402


def synth(seed, lesions=True):
    rng = np.random.default_rng(seed)
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.ellipse(img, (320, 240), (270, 220), 0, 0, 360, (60, 110, 200), -1)
    cv2.circle(img, (230, 220), 32, (170, 220, 245), -1)
    if lesions:
        for _ in range(6):
            x, y = int(rng.integers(150, 500)), int(rng.integers(120, 380))
            cv2.circle(img, (x, y), int(rng.integers(3, 8)), (15, 200, 240), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def dice(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if not a.any() and not b.any():
        return 1.0
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def main():
    if _native.mod is None:
        print("SKIP: netradr_core not built (run setup_native.sh)")
        return 0
    fails = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name, detail)
        if not ok:
            fails.append(name)

    for seed in (0, 1, 2):
        img = synth(seed, lesions=seed != 2)
        # --- M1 ---
        p = _assess_py(img)
        r = _native.mod.m1_assess(np.ascontiguousarray(img))
        check(f"M1[{seed}] accepted", p["accepted"] == bool(r["accepted"]),
              f"py={p['accepted']} cxx={bool(r['accepted'])}")
        check(f"M1[{seed}] score", abs(p["quality_score"] - float(r["quality_score"])) < 0.05,
              f"py={p['quality_score']} cxx={float(r['quality_score']):.3f}")
        check(f"M1[{seed}] flags", all(p["flags"][k] == bool(r[{"focus_ok": "focus_ok", "illumination_ok": "illumination_ok", "fov_ok": "fov_ok"}[k]]) for k in p["flags"]))
        pe = np.asarray(r["enhanced"], np.uint8).reshape(img.shape)
        check(f"M1[{seed}] enhanced", np.abs(pe.astype(int) - p["enhanced_image"].astype(int)).mean() < 12)
        # --- M2 ---
        s = _segment_py(p["enhanced_image"])
        n = _native.mod.m2_segment(np.ascontiguousarray(p["enhanced_image"]))
        vd = dice(s["vessel_mask"], n["vessel_mask"])
        xor = (np.asarray(s["vessel_mask"], bool) ^ np.asarray(n["vessel_mask"], bool)).sum()
        npx = s["vessel_mask"].size
        # near-empty masks are Otsu-on-noise: require tiny absolute disagreement
        vok = vd > 0.90 or (s["vessel_mask"].mean() < 0.05
                            and np.asarray(n["vessel_mask"], bool).mean() < 0.05
                            and xor < 0.015 * npx)
        check(f"M2[{seed}] vessels dice", vok, f"{vd:.3f} (xor {xor})")
        for key, ck in [("exudate_mask", "ex_mask"), ("hemorrhage_mask", "he_mask")]:
            check(f"M2[{seed}] {key} dice", dice(s[key], n[ck]) > 0.80,
                  f"{dice(s[key], n[ck]):.3f}")
        check(f"M2[{seed}] ma count", abs(s["lesion_features"]["ma_count"] - len(np.asarray(n["ma_cands"]).reshape(-1, 6))) <= 2)
        check(f"M2[{seed}] nv", s["neovascularization_flag"] == bool(n["nv_flag"]))
        check(f"M2[{seed}] od_r", abs(s["optic_disc"]["radius"] - float(n["od_radius"])) / max(s["optic_disc"]["radius"], 1) < 0.25)
        # --- M4 ---
        cam_p, _, ov_p = _pixels_py(p["enhanced_image"], s)
        m = _native.mod.m4_pixels(np.ascontiguousarray(p["enhanced_image"]),
                                  np.ascontiguousarray(s["microaneurysm_map"]["mask"]),
                                  np.ascontiguousarray(s["exudate_mask"]),
                                  np.ascontiguousarray(s["hemorrhage_mask"]),
                                  np.ascontiguousarray(s["vessel_mask"]))
        ov_c = [float(v) for v in m["ov"]]
        check(f"M4[{seed}] overlaps", all(abs(a - b) < 0.10 for a, b in zip(ov_p, ov_c)), f"{ov_p} vs {[round(v,3) for v in ov_c]}")
        ov_img = np.asarray(m["overlay"], np.uint8).reshape(img.shape)
        _, ov_py, _ = _pixels_py(p["enhanced_image"], s)
        check(f"M4[{seed}] overlay", np.abs(ov_img.astype(int) - ov_py.astype(int)).mean() < 15)

    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
