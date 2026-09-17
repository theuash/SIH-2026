# SIH 2026 PS 26038 — Build Plan (Python MVP mirroring MATLAB contracts)

> Stack: Python 3.14 + torch CPU + opencv + FastAPI backend, static `web/` frontend (no build step), simpy for M5.
> Data: zero local data at start — synthetic degradations + vendor pretrained weights first, APTOS/IDRiD/DRIVE fetch later.
> MATLAB port: every Python function mirrors its `.m` contract field-for-field, so port = rename + syntax.

## 1. Module reuse map (laziest that works)

| Module | Build | Vendor / reuse | Skipped (add when…) |
|---|---|---|---|
| M1 Quality | classical only: green-channel Laplacian variance (focus) + luminance histogram band (illumination) + dark-background FOV contour (crop/off-center). Enhance = CLAHE on L-channel + large-sigma Gaussian subtract + 3px median (preserves MA) | Thresholds from synthetically degraded images. Optional upgrade weight: `HzFu/EyeQ` MCF-Net DenseNet121 (~112MB, OneDrive). Enhancement alt: `joanshen0508/Fundus-correction-cofe-Net` | MCF-Net training — add when classical precision <85% |
| M2 vessels/OD/fovea | classical only: Frangi + top-hat + area-open vessels; brightest-component + HoughCircles OD; 2.5 OD-diameters temporal ROI, darkest-patch fovea | `scikit-image` frangi, `cv2.HoughCircles` | DRIVE U-Net retrain — add when Dice <0.7 |
| M2 lesions | inference-only, no training. 4 binary U-Nets | `apopli/diabetic-retinopathy` (IDRiD 1st fovea/5th EX) or `CaglarGuher/IDRiD-Eye-Fundus-Dataset-Lesion-Segmentation` (UNet++ VGG19, per-lesion `ma/he/ex/se` scripts) | Custom lesion training — add when AUPR gap proven |
| M2 MA sub-pixel | intensity-weighted centroid (≈2-line Gaussian-fit equivalent) | — | Full 2D Gaussian fit — add if PS explicitly graded |
| M3 grading | fork, don't train EfficientNet-B3 on CPU. Primary: `5seoyoung/dr-xai-fundus` (binary referable ≥2, EfficientNet-B0/B3, AUROC 0.985, CLAHE+circular mask already). Fallbacks: `Saksham52/dr_grading` (B4 + checkpoint included + Grad-CAM), `khir00/Diabetic-Retinopathy` (B3+CORAL, QWK 0.9143, best ordinal head) | `timm` EfficientNet + Ben Graham preprocess | Full APTOS train — add when fork QWK <0.85 on our split |
| M3 hybrid | weighted/logistic ensemble: CNN prob + lesion-feature XGB (`sklearn`). Keeps CNN-only / features-only / hybrid 3-way comparison free | `sklearn.ensemble` | Learned fusion layer — add when simple ensemble trails |
| M3 calibration | temperature scaling (~15 lines) on held-out val | — | — |
| M4 explain | `pytorch-grad-cam` lib, zero custom CAM. Lesion evidence = % overlap of each M2 mask with top-30% CAM pixels. Report = composed PNG via cv2/matplotlib | `pytorch-grad-cam`, `captum` optional | `mlreportgen` PDF — add when judges demand PDF |
| M5 sim | `simpy` discrete-event: arrival → upload (MB/Mbps) → process (measured s/img) → 30s review for referable only. Chart: wait vs #reviewers + "X cameras, Y servers, Z reviewers for 100k/yr" + bottleneck | `simpy` + matplotlib | `.slx`/SimEvents rebuild — map each queue to Queue+Server later |
| M6 pipeline | `run_pipeline.py`: folder → M1→M4 per image → `results.csv` + sklearn sens/spec/AUC/confusion | `sklearn.metrics` | App Designer port — after demo locked |

Frozen `lesion_features` schema (M2↔M3, single source `sih_dr/lesion_schema.py`):
`ma_count, ma_area, ex_count, ex_area, he_count, he_area, vessel_density, nv_present`

## 2. No-data strategy
1. Day-0 synthetic set: any 5 public fundus thumbs × degradations (Gaussian blur, brightness ±, vignette crop) → proves M1 separation + measures s/img for M5.
2. Pretrained weights carry accuracy; local run proves wiring + timing.
3. `data/splits.json` fixed up front (APTOS 80/20 stratified + all of IDRiD-B reserved) so later numbers are honest.

## 3. Build order (fastest to demo)
1. `lesion_schema.py` + stubs returning dummy-but-correct dicts → `run_pipeline.py` runs end-to-end empty.
2. M1 classical + M4 CAM wrapper (work with any M3 checkpoint).
3. Vendor M3 checkpoint + M2 lesion U-Net inference.
4. M5 sim + frontend + 3-way benchmark table.

## 4. Frontend (pitch-perfect, Apple-design) — `web/index.html` served by FastAPI `/`
Single static file, no npm. Streamlit/Gradio can't do springs / 1:1 drag / translucent chrome.

Flow (5-step rail, <30s review): Upload → Quality → Lesions → Grade → Explain. One screen, left→right stepper, symmetric enter/exit.
- Upload: drag-drop + paste, pointer-down highlight, instant thumb.
- Quality: translucent verdict chip over image (pass/recapture + plain-language reason).
- Lesions/Grade: mask toggle chips; grade dial = big display type (-0.02em tracking, tight leading) + calibrated confidence bar; low-confidence = warning style.
- Explain: CAM opacity slider (continuous during drag); lesion-evidence table ("2/3 MA overlap >70%"); report = bottom sheet spring (damping 1.0/response 0.35, interruptible, velocity handoff, rubber-band edge, drag-to-dismiss).
- Motion: `transform/opacity` only; default bounce 0, flick dismiss bounce 0.2. `prefers-reduced-motion` → cross-fade; `prefers-reduced-transparency` → solid chrome.
- M5 mini-view: wait-vs-reviewers chart + staffing line.
- Backend degrades to bundled sample + canned JSON when cold.

API (FastAPI, 5 routes): `POST /api/screen`, `GET /api/report/{id}`, `GET /api/sim`, `GET /api/metrics`, `GET /health`.

## 5. Validation (M6 claim)
CNN-only vs features-only vs hybrid on sealed APTOS test split: sens/spec/AUC for
referable DR (grade ≥2). Harness built (`sih_dr/splits.py`, `sih_dr/benchmark.py`,
`sih_dr/validate_seg.py`, `scripts/fetch_datasets.sh`) — smoke-tested on synthetic
data; real numbers pending dataset arrival. Arms without artifacts report `pending`,
never substituted numbers. Pitch = close-to-SOTA + explainable.

## 6. WhatsApp integration (built)
- Provider: Baileys sidecar `wa-gateway/` (QR pairing, free) at `WA_GATEWAY_URL`
  (default `http://127.0.0.1:3001`); `sih_dr/notify_whatsapp.py` abstracts
  `send_text/get_status/get_qr` so Cloud API can replace it without touching routes.
- Routes: `GET /api/whatsapp/qr|status`, `POST /api/notify` (consent-checked,
  EN/HI templates via `render()`), `POST /api/whatsapp/webhook` (STOP/HELP).
- Store: `data/patients.json` (phone, lang, consent_at, opted_out) +
  `data/notify_log.jsonl`. Gateway down → honest `queued-demo` mode, never blocks.
- Run gateway: `cd wa-gateway && npm i && node server.js`, scan QR via `GET /qr`.

## 7. Backend status (verified 2026-09-16, CPU-only, synthetic fundus)
- `sih_dr/`: `lesion_schema` + M1 classical gate + M2 classical seg (FOV-rim guard,
  bg-referenced MA confidence, sub-pixel centroids) + M3 heuristic hybrid
  (sqrt-compressed burden, T=1.3 calibration, torch-checkpoint hook ready) +
  M4 lesion-proximity CAM + overlap evidence + PNG report + M5 analytic M/M/c +
  `patients` + `notify_whatsapp` + `run_pipeline` (folder → results.csv + sens/spec/AUC).
- Smoke: lesion cartoon → grade 1 (MA-only, correct), clean → grade 0,
  confidences ~0.4 (honest); all 7 API routes live; `web/index.html` served at `/`.
- Known limits: thresholds tuned on cartoons — retune on APTOS/IDRiD; vendor
  U-Net + EfficientNet checkpoints not yet dropped into `models/`.
## 8. Repo layout
```
Plan.md
sih_dr/  m1_quality.py  m2_segment.py  m3_grade.py  m4_explain.py  m5_sim.py
         lesion_schema.py  run_pipeline.py  patients.py  notify_whatsapp.py
         _native.py + netradr_core*.so (built, gitignored)
native/  CMakeLists + include/ + src/ (m1/m2/m4/ops) + bindings.cpp
tests/test_parity.py  (32 C++-vs-Python checks)
web/index.html
app.py  (FastAPI: serves web/ + live /api/*)
wa-gateway/  (Baileys QR sidecar)
data/patients.json  data/notify_log.jsonl
```
## 9. Native C++ core (built, verified 2026-09-16)
- `native/` (C++17, zero third-party deps) + `setup_native.sh` → `sih_dr/netradr_core*.so`
  (gitignored, rebuilt per machine). M1/M2/M4 ported line-for-line (rect kernels —
  Python twins switched ellipse→rect to match); M3/M5 stay Python (nothing to gain).
- Tricks that mattered: 3-pass box ≈ big Gaussian (M1), 1/4-res heatmaps (M4),
  Lab LUTs, block Van Herk + AVX2 windowed min/max morphology, GIL released, zero-copy buffers.
- `tests/test_parity.py`: 32/32 green (Dice ~0.99 vessels, exact decisions).
  `NETRADR_IMPL=py` forces Python (MATLAB-port reference stays readable).
- Honest ceiling: M2 single-image 0.7x of SIMD-tuned OpenCV (ours 58ms vs 40ms) —
  end-to-end still wins 2.9–4x via M1 (4.9x) + threading. GPU path skipped (CPU target).

