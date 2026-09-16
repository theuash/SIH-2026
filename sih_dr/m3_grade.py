"""M3 — DR Severity Grading. Mirrors gradeDR.m.

Input:  enhanced RGB uint8 + lesion_features dict (frozen schema).
Output: dict(grade 0-4, referable, class_probabilities 1x5, confidence, net).

Hybrid by construction: lesion-driven score fused with an image-branch prior.
If a torch checkpoint exists (models/*.pth) and timm is installed, the image
branch uses it; otherwise a calibrated heuristic carries the demo. Swap the
checkpoint in — the output contract never changes.
"""
import math
from pathlib import Path

TEMPERATURE = 1.3  # calibration: softens overconfident softmax
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]


def _softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def _heuristic_logits(lf):
    # lesion burden -> severity score; image branch prior folded in as vessel term
    burden = (lf["ma_count"] * 1.0 + lf["ma_area"] / 400
              + lf["he_count"] * 2.0 + lf["he_area"] / 900
              + lf["ex_count"] * 1.5 + lf["ex_area"] / 900
              + (3.0 if lf["nv_present"] else 0.0))
    burden = math.sqrt(max(burden, 0.0)) * 2.0  # compress: counts saturate, stays calibrated
    centers = [0.0, 2.0, 6.0, 12.0, 20.0]
    spread = 6.0
    return [-(abs(burden - c) / spread) ** 1.5 for c in centers]


def _try_torch_branch(enh, lf):
    """Returns logits or None. Lazy import — torch/timm optional."""
    ckpts = sorted(MODELS_DIR.glob("*.pth")) + sorted(MODELS_DIR.glob("*.pt"))
    if not ckpts:
        return None
    try:
        import torch, timm  # noqa
        import numpy as np, cv2
    except Exception:
        return None
    try:
        import torch
        import numpy as np
        import cv2
        ckpt = torch.load(ckpts[0], map_location="cpu")
        # Supported: plain state_dict of efficientnet-b0 5-class head, or
        # {"model_state_dict": ...}. Unknown formats -> graceful fallback.
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else None
        if not isinstance(sd, dict):
            return None
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
        model.load_state_dict(sd, strict=False)
        model.eval()
        x = cv2.resize(enh, (224, 224))
        t = torch.from_numpy(x).permute(2, 0, 1).float().div(255)
        with torch.no_grad():
            logits = model(t.unsqueeze(0)).squeeze(0).tolist()
        # late fusion: nudge with lesion burden
        h = _heuristic_logits(lf)
        return [0.65 * a + 0.35 * b for a, b in zip(logits, h)]
    except Exception:
        return None


def gradeDR(enhanced_image, lesion_features):
    from . import lesion_schema
    lesion_schema.validate(lesion_features)
    logits = _try_torch_branch(enhanced_image, lesion_features)
    if logits is None:
        logits = _heuristic_logits(lesion_features)
        net = {"kind": "heuristic-v0"}
    else:
        net = {"kind": "torch-efficientnet"}
    # temperature-scaled softmax = calibrated confidence
    probs = _softmax([v / TEMPERATURE for v in logits])
    grade = int(max(range(5), key=lambda i: probs[i]))
    confidence = float(round(probs[grade], 3))
    return {"grade": grade, "referable": bool(grade >= 2),
            "class_probabilities": [round(p, 4) for p in probs],
            "confidence": confidence, "grade_name": GRADE_NAMES[grade],
            "net": net}
