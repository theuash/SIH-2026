"""M2 — Structure Segmentation. Mirrors segmentStructures.m.

Input:  enhanced RGB uint8 (M1 output).
Output: dict(vessel_mask, optic_disc{mask,centroid,radius}, fovea_centroid,
             microaneurysm_map{mask,candidates}, exudate_mask, hemorrhage_mask,
             neovascularization_flag, nv_regions, lesion_features)
Classical only. Vendor U-Net weights plug in here later (same output shape).
"""
import cv2
import math
import numpy as np
from . import lesion_schema
from . import _native


def _area_open(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = True
    return out


def _weighted_centroids(mask, gray, bg):
    """Intensity-weighted (sub-pixel) centroids per component.
    bg = retinal background median; confidence = darkness vs background."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        w = 255 - gray[ys, xs].astype(float) + 1  # darker = heavier (red lesions)
        area = len(xs)
        cx = float((xs * w).sum() / w.sum())
        cy = float((ys * w).sum() / w.sum())
        conf = float(np.clip((float(bg) - float(gray[ys, xs].mean())) / 60, 0, 1))
        out.append({"x": float(cents[i][0]), "y": float(cents[i][1]),
                    "x_subpixel": cx, "y_subpixel": cy,
                    "area": int(area), "confidence": round(conf, 3)})
    return out


def _segment_py(enh):
    assert enh.dtype == np.uint8 and enh.ndim == 3
    h, w = enh.shape[:2]
    scale = (h * w) / (512 * 512)  # size-normalize area thresholds
    green = enh[:, :, 1]
    lab = cv2.cvtColor(enh, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    # interior: eroded retinal mask — FOV rim/top-hat edge artifacts live here
    interior = cv2.erode((L > 12).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))).astype(bool)

    # --- vessels: top-hat on CLAHE green + length filter ---
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(green)
    tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    _, v = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessel_mask = _area_open(v > 0, int(60 * scale))
    vessel_density = float(vessel_mask.mean())

    # --- optic disc: brightest large red component + enclosing circle ---
    red = enh[:, :, 0].astype(float)
    thr = np.percentile(red, 99.2)
    bright = _area_open(red > thr, int(800 * scale))
    od_mask = np.zeros((h, w), bool)
    od_c, od_r = [w * 0.35, h * 0.5], 0.0
    if bright.any():
        n, labm, stats, cents = cv2.connectedComponentsWithStats(bright.astype(np.uint8), 8)
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = (labm == i)
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
            od_c, od_r = [float(x), float(y)], float(r)
            tmp = np.zeros((h, w), np.uint8)
            cv2.circle(tmp, (int(x), int(y)), int(r), 1, -1)
            od_mask = tmp > 0
    optic_disc = {"mask": od_mask, "centroid": od_c, "radius": od_r}

    # --- fovea: darkest patch ~2.5 OD-diameters temporal to OD ---
    dd = max(od_r * 2, 20)
    # temporal side = darker half mean; fallback left
    left, right = green[:, :w // 2].mean(), green[:, w // 2:].mean()
    direction = -1 if left < right else 1
    fx = float(np.clip(od_c[0] + direction * 2.5 * dd, dd, w - dd))
    fy = float(np.clip(od_c[1], dd, h - dd))
    roi = green[int(max(fy - dd, 0)):int(min(fy + dd, h)),
                int(max(fx - dd, 0)):int(min(fx + dd, w))]
    if roi.size:
        _, _, minloc, _ = cv2.minMaxLoc(cv2.GaussianBlur(roi, (15, 15), 0))
        fx = float(max(fx - dd, 0) + minloc[0])
        fy = float(max(fy - dd, 0) + minloc[1])

    # --- exudates: bright in L + yellow in b, outside dilated OD ---
    b = lab[:, :, 2].astype(float)
    od_dil = cv2.dilate(od_mask.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))).astype(bool)
    ex = (L > np.percentile(L, 97.5)) & (b > np.percentile(b, 75)) & (~od_dil) & interior
    exudate_mask = _area_open(ex, int(25 * scale))

    # --- dark lesions: small top-hat (MA) vs larger blobs (HE) ---
    small = cv2.morphologyEx(255 - green, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    _, ma_raw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ma_raw = (ma_raw > 0) & (~od_dil) & (~vessel_mask) & interior
    big = cv2.morphologyEx(255 - green, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    _, he_raw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    he_raw = (he_raw > 0) & (~od_dil) & interior

    ma_mask = _area_open(ma_raw, int(3 * scale)) & ~_area_open(ma_raw, int(220 * scale))
    # MA = small components; HE = large dark components minus vessels
    lbl, labh, statsh, _ = cv2.connectedComponentsWithStats(he_raw.astype(np.uint8), 8)
    he_mask = np.zeros((h, w), bool)
    for i in range(1, lbl):
        if statsh[i, cv2.CC_STAT_AREA] >= int(220 * scale):
            he_mask[labh == i] = True
    he_mask = he_mask & ~vessel_mask

    ma_cands = [c for c in _weighted_centroids(ma_mask, green, np.median(green[interior]))
                if c["confidence"] >= 0.2]
    he_cands = [c for c in _weighted_centroids(he_mask, green, np.median(green[interior]))
                if c["confidence"] >= 0.2]

    # --- neovascularization heuristic: dense fine vessels near OD ---
    ring = cv2.dilate(od_mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))).astype(bool) & ~od_mask
    ring_density = float(vessel_mask[ring].mean()) if ring.any() else 0.0
    nv_flag = bool(ring_density > 0.18 and (len(ma_cands) + len(he_cands) > 4))
    nv_regions = (ring & vessel_mask) if nv_flag else np.zeros((h, w), bool)

    def _stats(mask, cands):
        return len(cands), int(mask.sum())

    ma_n, ma_a = len(ma_cands), sum(c["area"] for c in ma_cands)
    he_n, he_a = len(he_cands), sum(c["area"] for c in he_cands)
    n_ex, labex, statsex, _ = cv2.connectedComponentsWithStats(exudate_mask.astype(np.uint8), 8)
    ex_n, ex_a = max(n_ex - 1, 0), int(exudate_mask.sum())

    lesion_features = {"ma_count": ma_n, "ma_area": ma_a, "ex_count": ex_n,
                       "ex_area": ex_a, "he_count": he_n, "he_area": he_a,
                       "vessel_density": round(vessel_density, 4),
                       "nv_present": nv_flag}
    lesion_schema.validate(lesion_features)

    return {"vessel_mask": vessel_mask, "optic_disc": optic_disc,
            "fovea_centroid": [fx, fy],
            "microaneurysm_map": {"mask": ma_mask, "candidates": ma_cands},
            "exudate_mask": exudate_mask, "hemorrhage_mask": he_mask,
            "neovascularization_flag": nv_flag, "nv_regions": nv_regions,
            "lesion_features": lesion_features}


def _cands_from_np(arr):
    out = []
    for row in np.asarray(arr, dtype=float).reshape(-1, 6):
        x, y, xs, ys, area, conf = (float(v) for v in row)
        out.append({"x": x, "y": y, "confidence": round(conf, 3),
                    "x_subpixel": xs, "y_subpixel": ys, "area": int(area)})
    return out


def segmentStructures(enh):
    """Dispatch: C++ fast path, Python twin fallback. Same contract."""
    assert enh.dtype == np.uint8 and enh.ndim == 3
    if _native.want_native():
        r = _native.mod.m2_segment(np.ascontiguousarray(enh))
        ma_cands = _cands_from_np(r["ma_cands"])
        he_cands = _cands_from_np(r["he_cands"])
        nv_flag = bool(r["nv_flag"])
        lesion_features = {
            "ma_count": len(ma_cands), "ma_area": int(r["ma_area"]),
            "ex_count": int(r["ex_count"]), "ex_area": int(r["ex_area"]),
            "he_count": len(he_cands), "he_area": int(r["he_area"]),
            "vessel_density": round(float(r["vessel_density"]), 4),
            "nv_present": nv_flag}
        lesion_schema.validate(lesion_features)
        return {
            "vessel_mask": np.array(r["vessel_mask"], dtype=bool),
            "optic_disc": {"mask": np.array(r["od_mask"], dtype=bool),
                           "centroid": [float(r["od_centroid"][0]),
                                        float(r["od_centroid"][1])],
                           "radius": float(r["od_radius"])},
            "fovea_centroid": [float(r["fovea"][0]), float(r["fovea"][1])],
            "microaneurysm_map": {"mask": np.array(r["ma_mask"], dtype=bool),
                                  "candidates": ma_cands},
            "exudate_mask": np.array(r["ex_mask"], dtype=bool),
            "hemorrhage_mask": np.array(r["he_mask"], dtype=bool),
            "neovascularization_flag": nv_flag,
            "nv_regions": np.array(r["nv_regions"], dtype=bool),
            "lesion_features": lesion_features}
    return _segment_py(enh)
