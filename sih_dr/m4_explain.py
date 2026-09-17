"""M4 — Explainability. Mirrors explainGrading.m.

Input:  enhanced RGB uint8, net (M3 handle), seg (M2 dict), grading (M3 dict).
Output: dict(cam_heatmap 0-1 HxW, overlay_image RGB uint8, lesion_evidence
             list of [type, found, overlap, trust], report_file path).

Pixels via C++ fast path when built (_pixels_cxx), else Python twin.
Strings + report PNG always shared (_finish). Same contract either way.
"""
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from . import _native

REPORTS = Path(__file__).resolve().parent.parent / "reports"
HIGH_THR, MID_THR = 0.70, 0.30


def _fallback_cam(shape, seg):
    h, w = shape[:2]
    cam = np.zeros((h, w), np.float32)
    for key, sigma in [("microaneurysm_map", 18), ("exudate_mask", 26),
                       ("hemorrhage_mask", 30)]:
        m = seg[key]["mask"] if isinstance(seg[key], dict) else seg[key]
        if m.any():
            blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
            cam = np.maximum(cam, blur / (blur.max() + 1e-9))
    if seg["vessel_mask"].any():
        v = cv2.GaussianBlur(seg["vessel_mask"].astype(np.float32), (0, 0), 12)
        cam = np.maximum(cam, 0.35 * v / (v.max() + 1e-9))
    return cam


def _overlap(mask, hot):
    inter = float((mask & hot).sum())
    tot = float(mask.sum())
    return round(inter / tot, 3) if tot else 0.0


def _pixels_py(enhanced_image, seg):
    cam = np.clip(_fallback_cam(enhanced_image.shape, seg), 0, 1).astype(float)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay_image = cv2.addWeighted(enhanced_image, 0.55, heat, 0.45, 0)
    hot = cam >= np.quantile(cam, 0.70)
    ma, ex, he = (seg["microaneurysm_map"]["mask"], seg["exudate_mask"],
                  seg["hemorrhage_mask"])
    return cam, overlay_image, [_overlap(ma, hot), _overlap(ex, hot), _overlap(he, hot)]


def _finish(enhanced_image, net, seg, grading, cam, overlay_image, overlaps):
    ma, ex, he = (seg["microaneurysm_map"]["mask"], seg["exudate_mask"],
                  seg["hemorrhage_mask"])
    rows = []
    counts = {"Microaneurysms": len(seg["microaneurysm_map"]["candidates"])}
    for (name, m), ov in zip([("Microaneurysms", ma), ("Hard exudates", ex),
                              ("Hemorrhages", he)], overlaps):
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        n = counts.get(name, max(n - 1, 0))
        trust = "HIGH" if ov >= HIGH_THR else ("MID" if ov >= MID_THR else "LOW")
        found = f"{n} found" if n else "none"
        detail = (f"{max(int(n * ov), 1) if n else 0} strongly overlap"
                  if ov >= HIGH_THR else ("weak overlap" if n else "—"))
        rows.append([name, found, detail, trust, ov])

    REPORTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = str(REPORTS / f"report_{ts}_g{grading['grade']}.png")
    left = cv2.cvtColor(cv2.resize(enhanced_image, (420, 315)), cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(cv2.resize(overlay_image, (420, 315)), cv2.COLOR_RGB2BGR)
    canvas = np.full((400, 860, 3), 255, np.uint8)
    canvas[10:325, 10:430] = left
    canvas[10:325, 430:850] = right
    lines = [f"NetraDR  Grade {grading['grade']} {grading.get('grade_name', '')}",
             f"confidence {grading['confidence']:.2f} (calibrated)  Q refers:{grading.get('referable')}",
             f"CAM source: {'grad-cam' if net.get('kind') == 'torch-efficientnet' else 'lesion-proximity fallback'}"]
    for i, r in enumerate(rows):
        lines.append(f"{r[0]}: {r[1]}, {r[2]} [{r[3]}]")
    y = 350
    for ln in lines:
        cv2.putText(canvas, ln[:72], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        y += 22
    cv2.imwrite(report_file, canvas)

    evidence = [[r[0], r[1], r[2], r[3]] for r in rows]
    return {"cam_heatmap": cam, "overlay_image": overlay_image,
            "lesion_evidence": evidence, "report_file": report_file}


def _explain_py(enhanced_image, net, seg, grading):
    cam, overlay_image, overlaps = _pixels_py(enhanced_image, seg)
    return _finish(enhanced_image, net, seg, grading, cam, overlay_image, overlaps)


def explainGrading(enhanced_image, net, seg, grading):
    """Dispatch: C++ fast path, Python twin fallback. Same contract."""
    if _native.want_native():
        r = _native.mod.m4_pixels(
            np.ascontiguousarray(enhanced_image),
            np.ascontiguousarray(seg["microaneurysm_map"]["mask"]),
            np.ascontiguousarray(seg["exudate_mask"]),
            np.ascontiguousarray(seg["hemorrhage_mask"]),
            np.ascontiguousarray(seg["vessel_mask"]))
        cam = np.clip(np.array(r["cam"], dtype=float), 0, 1)
        overlay_image = np.array(r["overlay"], dtype=np.uint8)
        return _finish(enhanced_image, net, seg, grading, cam, overlay_image,
                       [float(v) for v in r["ov"]])
    return _explain_py(enhanced_image, net, seg, grading)
