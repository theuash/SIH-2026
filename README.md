# NetraDR — Explainable DR Screening (SIH 2026 · PS 26038)

Fundus photo in → quality gate → lesions → grade with calibrated confidence →
Grad-CAM + lesion evidence → patient reached on WhatsApp. Human-in-the-loop, never auto-diagnosis.

## Run (2 minutes)

```bash
pip install --break-system-packages -r requirements.txt   # needs Python 3.10+
python app.py                                             # http://127.0.0.1:8000
```

Open the URL, hit **Sample retina → Screen this eye**. No downloads, no data needed.

## Other uses

```bash
# batch a folder -> results.csv (+ sens/spec/AUC with labels.csv: file,grade)
python -m sih_dr.run_pipeline --input /path/to/images --out results.csv --labels labels.csv

# district capacity model
python -c "from sih_dr.m5_sim import runScreeningSim; print(runScreeningSim(n_reviewers=3))"
```

## WhatsApp (optional)

```bash
cd wa-gateway && npm i && node server.js   # :3001, scan QR via the app's pairing card
```

Without it the app still works — messages queue in `demo` mode and log to
`data/notify_log.jsonl`. Consent + STOP/HELP opt-out enforced in `sih_dr/patients.py`.

## Layout

```
web/index.html        # pitch frontend (static, works even via file://)
app.py                # FastAPI: serves web/ + /api/screen /notify /whatsapp/* /sim /metrics
sih_dr/               # M1 quality, M2 segmentation, M3 grading, M4 explainability,
                      # M5 sim, M6 pipeline, patients, WhatsApp notify
wa-gateway/           # Baileys QR-pairing sidecar (Node)
Plan.md               # build plan + module reuse map
files/                # original SIH briefs + per-member contexts
```

## Drop in real models (no code change)

- `models/*.pth` — EfficientNet-B0 5-class state dict → M3 image branch activates automatically
- IDRiD U-Net lesion weights → wire into `sih_dr/m2_segment.py` (same output shape)

## Limits (read before judging)

Thresholds tuned on synthetic images — retune on APTOS/IDRiD before quoting
metrics. Heuristic M3 is a wiring stand-in; the hybrid claim needs the checkpoints above.
