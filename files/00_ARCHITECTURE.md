# System Architecture — SIH 2026 PS 26038
## Explainable AI for Diabetic Retinopathy Screening in Rural India

## 1. Pipeline Overview

```
[Fundus Camera Image]
        │
        ▼
┌───────────────────────────┐
│ M1. Quality Assessment &  │  reject → recapture feedback to operator
│     Enhancement            │
└─────────────┬─────────────┘
              │ enhanced_image (accepted only)
              ▼
┌───────────────────────────┐
│ M2. Structure Segmentation │
│  vessels / OD / fovea /    │
│  MA / EX / HE / NV         │
└─────────────┬─────────────┘
              │ struct masks + lesion features
              ▼
┌───────────────────────────┐
│ M3. DR Severity Grading    │  CNN (image) + lesion features (hybrid)
│  Grade 0-4, referable flag │
└─────────────┬─────────────┘
              │ grade + probabilities + trained net
              ▼
┌───────────────────────────┐
│ M4. Explainability         │  Grad-CAM + lesion evidence + report
└─────────────┬─────────────┘
              │ annotated PDF/report
              ▼
   Ophthalmologist review (<30s) ── confirm / override
              │
              ▼
┌───────────────────────────┐
│ M5. Simulink Workflow      │  runs in parallel, not per-image —
│  Sim (throughput, capacity)│  models the district-scale system
└───────────────────────────┘

   M6. Integration + Validation + Demo (ties everything together)
```

## 2. Module Interface Contracts

Each module is a MATLAB function (or small package) with a fixed input/output
contract so all 6 people can build independently and plug together at the end.
Treat these signatures as the source of truth — do not change without telling
the whole team.

### M1 — `assessAndEnhance(img)`
- **Input:** `img` — raw RGB fundus image (uint8, any resolution)
- **Output:** `struct` with fields:
  - `accepted` (logical)
  - `quality_score` (0–1 double)
  - `flags` — struct: `focus_ok`, `illumination_ok`, `fov_ok` (logical each)
  - `enhanced_image` (uint8 RGB, same size, CLAHE + illumination-normalized + denoised) — only populated if accepted or borderline
  - `feedback_msg` (string) — human-readable recapture reason if rejected

### M2 — `segmentStructures(enhanced_image)`
- **Input:** `enhanced_image` from M1
- **Output:** `struct` with fields:
  - `vessel_mask` (logical, same size)
  - `optic_disc` — struct: `mask`, `centroid [x y]`, `radius`
  - `fovea_centroid` `[x y]`
  - `microaneurysm_map` — logical mask + `candidates` table (x, y, confidence, sub-pixel-refined [x y])
  - `exudate_mask` (logical)
  - `hemorrhage_mask` (logical)
  - `neovascularization_flag` (logical) + `nv_regions` (mask)
  - `lesion_features` — table: counts/areas per lesion type (this feeds M3)

### M3 — `gradeDR(enhanced_image, lesion_features)`
- **Input:** enhanced image from M1 + `lesion_features` table from M2
- **Output:** `struct`:
  - `grade` (0–4 int, ICDR scale)
  - `referable` (logical, grade ≥ 2)
  - `class_probabilities` (1x5 double, softmax)
  - `confidence` (0–1, calibrated — not raw softmax max)
  - `net` handle (trained dlnetwork, needed by M4)

### M4 — `explainGrading(enhanced_image, net, seg_struct, grading_struct)`
- **Input:** everything from M1–M3
- **Output:** `struct`:
  - `cam_heatmap` (double, normalized 0–1, same size as image)
  - `overlay_image` (uint8, heatmap blended on image)
  - `lesion_evidence` — table: which detected lesions overlap high-CAM regions, with % overlap
  - `report_file` (path to generated PDF/PNG annotated report)

### M5 — Simulink model (not a function; a `.slx` file + a MATLAB driver script `runScreeningSim.m`)
- **Inputs (as workspace variables / Simulink parameters):** acquisition rate (images/hr/center), image size (MB), network bandwidth (Mbps), per-image processing time (from real timing of M1–M4 on the target hardware), ophthalmologist review time (~30s per flagged case), number of centers, number of reviewing ophthalmologists.
- **Output:** throughput report — queue lengths, bottleneck stage, images/day capacity, recommended number of ophthalmologists/servers for a target district (100,000 patients/year ⇒ ~275/day average, plan for peak factor).

### M6 — Integration
- A single driver (`runPipeline.m` or MATLAB App Designer app) that calls M1→M2→M3→M4 in sequence on a folder of images, logs results to a table, and computes validation metrics (sensitivity, specificity, AUC, confusion matrix) against dataset ground-truth labels.

## 3. Why this order (dependency chain)
M1 has no dependencies → can start immediately with any dataset.
M2 depends on M1's output format only (can start with raw images + assumed enhancement stub while M1 is built).
M3 depends on M2's `lesion_features` schema — agree on this table's columns in week 1 and freeze it.
M4 depends on M3's trained network object.
M5 is independent of the algorithmic modules — only needs *timing numbers*, which can be stubbed/estimated early and refined later.
M6 depends on all of the above but can be scaffolded from day 1 using stub functions that return dummy structs matching the contracts above.

## 4. Suggested tech per module
| Module | Primary Toolbox | Key MATLAB functions to look up |
|---|---|---|
| M1 | Image Processing Toolbox | `adapthisteq`, `imgaussfilt`, `imnlmfilt`, `fspecial`, `imbinarize`, `regionprops` |
| M2 | Image Processing + Computer Vision + Deep Learning Toolbox | `imtophat`, `imreconstruct`, `bwareaopen`, `imfindcircles`, `trainNetwork`/`unetLayers`, `activecontour` |
| M3 | Deep Learning Toolbox + Statistics and ML Toolbox | `imagePretrainedNetwork` (ResNet/EfficientNet), `trainnet`, `fitPosterior`/temperature scaling for calibration |
| M4 | Deep Learning Toolbox | `gradCAM`, `imoverlay`, MATLAB Report Generator (`mlreportgen.dom`) |
| M5 | Simulink, SimEvents (if licensed) | queueing blocks, `Delay`, `Queue`, or a discrete-event MATLAB script if SimEvents unavailable |
| M6 | Statistics and ML Toolbox | `perfcurve` (ROC/AUC), `confusionmat`, App Designer |

## 5. Validation strategy (for the "outperforms single-technique" claim)
Run three configurations and compare on a held-out split of APTOS + IDRiD:
1. **CNN-only** grading (M3 image branch alone)
2. **Hand-crafted-features-only** grading (lesion counts/areas → classifier)
3. **Integrated pipeline** (M1 enhancement + M2 features + M3 hybrid + calibration)
Report sensitivity/specificity/AUC for referable DR (grade ≥2) for all three — the pitch is that (3) beats both (1) and (2) alone, and is explainable while (1) is not.
