"""M1 — Quality Assessment & Enhancement. Mirrors assessAndEnhance.m.

Input:  img RGB uint8 HxWx3, any resolution.
Output: dict(accepted, quality_score, flags{focus_ok, illumination_ok, fov_ok},
             enhanced_image, feedback_msg)
Classical only (no DL): Laplacian variance + luminance band + FOV contour.
"""
import cv2
import numpy as np

from . import _native

# Thresholds tuned on synthetically degraded fundus thumbs; retune on EyeQ/APTOS.
FOCUS_THR = 28.0          # Laplacian variance on green channel
DARK_FRAC_THR = 0.25      # allowed underexposed retinal fraction
BRIGHT_FRAC_THR = 0.12    # allowed overexposed retinal fraction
FOV_COVER_THR = 0.55      # retinal mask must cover >=55% of frame
FOV_OFF_THR = 0.28        # FOV center offset <=28% of width


def _retinal_mask(v):
    m = (v > 12).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    return m > 0


def _feedback(accepted, focus_ok, illumination_ok, fov_ok):
    if accepted:
        return "Accept."
    reasons = []
    if not focus_ok:
        reasons.append("image is blurred — hold the camera steady and refocus")
    if not illumination_ok:
        reasons.append("lighting is off (too dark/bright) — adjust illumination")
    if not fov_ok:
        reasons.append("retina not fully visible — recenter so the full eye fills the frame")
    return "Recapture: " + "; ".join(reasons) + "."


def _assess_py(img):
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, \
        "img must be RGB uint8 HxWx3"
    h, w = img.shape[:2]
    green = img[:, :, 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2]
    mask = _retinal_mask(val)
    retina = green[mask] if mask.any() else green.ravel()

    # 1. focus: variance of Laplacian on green
    lap_var = float(cv2.Laplacian(green, cv2.CV_64F).var())
    focus_ok = bool(lap_var >= FOCUS_THR)

    # 2. illumination: underexposed (<30) / overexposed (>235) fractions
    dark_frac = float((retina < 30).mean())
    bright_frac = float((retina > 235).mean())
    mean_v = float(retina.mean())
    illumination_ok = bool(dark_frac <= DARK_FRAC_THR
                           and bright_frac <= BRIGHT_FRAC_THR
                           and 35 <= mean_v <= 225)

    # 3. field of view: largest contour coverage + centering
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c) / (h * w)
        M = cv2.moments(c)
        cx = M["m10"] / (M["m00"] + 1e-9)
        off = abs(cx - w / 2) / w
        fov_ok = bool(area >= FOV_COVER_THR and off <= FOV_OFF_THR)
    else:
        area, off, fov_ok = 0.0, 1.0, False

    # combined score: worst-flag-driven, 0..1
    parts = [min(lap_var / (FOCUS_THR * 3), 1.0),
             1.0 - min((dark_frac + bright_frac) / 0.5, 1.0),
             min(area / 0.85, 1.0)]
    quality_score = float(round(sum(parts) / 3, 3))
    accepted = bool(focus_ok and illumination_ok and fov_ok)
    feedback_msg = _feedback(accepted, focus_ok, illumination_ok, fov_ok)

    enhanced_image = _enhance(img)  # always populated; harmless on good images
    return {"accepted": accepted, "quality_score": quality_score,
            "flags": {"focus_ok": focus_ok, "illumination_ok": illumination_ok,
                      "fov_ok": fov_ok},
            "enhanced_image": enhanced_image, "feedback_msg": feedback_msg}


def assessAndEnhance(img):
    """Dispatch: C++ fast path, Python twin fallback. Same contract."""
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, \
        "img must be RGB uint8 HxWx3"
    if _native.want_native():
        r = _native.mod.m1_assess(np.ascontiguousarray(img))
        flags = {"focus_ok": bool(r["focus_ok"]),
                 "illumination_ok": bool(r["illumination_ok"]),
                 "fov_ok": bool(r["fov_ok"])}
        accepted = bool(r["accepted"])
        return {"accepted": accepted,
                "quality_score": float(round(float(r["quality_score"]), 3)),
                "flags": flags,
                "enhanced_image": np.array(r["enhanced"], dtype=np.uint8),
                "feedback_msg": _feedback(accepted, **flags)}
    return _assess_py(img)


def _enhance(img):
    # CLAHE on L + illumination flatten + edge-preserving denoise (MA-safe)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    # flatten uneven lighting: subtract large-sigma background, re-add base
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=min(l.shape) / 8)
    flat = cv2.addWeighted(l, 1.15, bg, -0.15, 8)
    enh = cv2.cvtColor(cv2.merge([flat, a, b]), cv2.COLOR_LAB2RGB)
    enh = cv2.bilateralFilter(enh, 3, 25, 25)  # 3px: denoises, keeps microaneurysms
    return enh
