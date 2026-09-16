# MASTER CONTEXT — SIH 2026 PS 26038
## Explainable AI for Diabetic Retinopathy Screening in Rural India

This is the single source of truth for the whole team and project.
Use this file when you need the full picture (e.g. team lead's AI,
onboarding a new member, or preparing the PPT/report). Individual
members should still use their own `context_memberN_*.md` file
day-to-day so their AI assistant isn't overloaded with irrelevant detail.

---

## Project Background (share this with your AI assistant first)

We are building a prototype for **SIH 2026 Problem Statement 26038**:
"Explainable AI for Diabetic Retinopathy Screening in Rural India."

Context: India has 77M+ diabetic adults; ~18% develop diabetic retinopathy
(DR), a leading cause of preventable blindness, but there is only ~1
ophthalmologist per 100,000 rural population. We are building a MATLAB-based
pipeline that screens fundus images for DR, grades severity on the
International Clinical DR (ICDR) scale (0-4), and explains its reasoning
(Grad-CAM + lesion evidence) so an ophthalmologist can confirm a case in
under 30 seconds (human-in-the-loop, not full automation).

The system has 6 modules built by 6 different team members, each working
with their own AI assistant, and wired together at the end. This master file
covers the full system; each member also has a `context_memberN_*.md` file
scoped to just their module's contract and task detail. The interface
contracts below (Part 1) are the source of truth — nobody should change one
without telling the whole team, since other modules depend on it.

Tools available: MATLAB Image Processing Toolbox, Computer Vision Toolbox,
Deep Learning Toolbox, Medical Imaging Toolbox, Simulink, Statistics and
Machine Learning Toolbox.

Target metrics for the *overall* system (not just your module): >90%
sensitivity and >85% specificity for referable DR (grade >=2).

---

# PART 1 — SYSTEM ARCHITECTURE


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

---

# PART 2 — TEAM PLAN

Each member owns one module end-to-end, works with their own AI assistant using
their dedicated context file (`context_memberN_*.md`), and commits to the shared
interface contracts in `00_ARCHITECTURE.md`.

| # | Member | Module | Context file | Core deliverable |
|---|---|---|---|---|
| 1 | M1 owner | Quality Assessment & Enhancement | `context_member1_image_quality.md` | `assessAndEnhance.m` + demo on 20 good/bad images |
| 2 | M2 owner | Structure Segmentation | `context_member2_segmentation.md` | `segmentStructures.m`, validated on IDRiD/DRIVE ground truth |
| 3 | M3 owner | DR Severity Grading | `context_member3_grading.md` | trained network + `gradeDR.m`, sensitivity/specificity report |
| 4 | M4 owner | Explainability | `context_member4_explainability.md` | `explainGrading.m` + sample annotated reports |
| 5 | M5 owner | Simulink Workflow Simulation | `context_member5_simulink.md` | `.slx` model + resource-allocation recommendation |
| 6 | M6 owner (team lead / integrator) | Integration, Validation, Demo, PPT | `context_member6_integration.md` | end-to-end demo app + benchmark comparison table + final deck |

Reassign these based on actual skills — e.g. whoever is strongest at deep
learning should take M3, whoever is best with Simulink/systems thinking
should take M5. The important thing is everyone commits to the interface
contract, not who does which module.

## Suggested 3-week timeline (adapt to your actual SIH internal deadlines)

**Week 1 — Foundations**
- All: download & explore datasets (see `00_DATASETS_GUIDE.md`)
- M1–M5: build a stub version of your function that returns dummy but
  correctly-shaped output matching the contract, so M6 can wire up the
  full pipeline immediately
- M6: scaffold `runPipeline.m` using the 5 stubs; set up the shared repo/folder
  structure and a results-logging table
- Freeze the `lesion_features` table schema (M2↔M3 boundary) — this is the
  one interface most likely to cause pain if left loose

**Week 2 — Real implementation**
- M1: real quality metrics + CLAHE/denoising, tested against deliberately
  degraded images
- M2: real segmentation for each structure, validated against IDRiD/DRIVE masks
- M3: train the CNN + hybrid classifier on APTOS (+ IDRiD grading labels),
  get sensitivity/specificity numbers
- M4: Grad-CAM wired to M3's trained network; lesion-overlap evidence table
- M5: build Simulink model with placeholder timing numbers from M1–M4's
  early runs
- M6: keep re-running the integrated pipeline daily as real modules replace stubs;
  start the benchmark-comparison table (integrated vs CNN-only vs features-only)

**Week 3 — Integration, validation, packaging**
- Full pipeline run on held-out test split, compute final sensitivity/specificity/AUC
- M5: plug in real timing numbers, finalize resource-allocation recommendation
  for a 100,000-patient/year district
- M4: polish annotated report format, get informal "would an ophthalmologist
  find this useful in <30s" feedback from anyone with clinical contacts
- M6: build the demo (App Designer or Live Script walkthrough), assemble PPT,
  write up validation results vs published APTOS/IDRiD benchmarks
- All: rehearse demo, prep for Q&A on clinical validity, explainability, and
  why the integrated pipeline beats single-technique baselines

## Working with AI per-module
Give your AI assistant (Claude, or whichever you use) the shared master
section at the top of your `context_memberN_*.md` file first, then your
module's task section. Keep the interface contract (inputs/outputs) pinned
in every session — that's what keeps all 6 people's code compatible without
needing to coordinate constantly.

---

# PART 3 — DATASETS

| Dataset | Access | What it's for | Who needs it |
|---|---|---|---|
| **APTOS 2019** | Kaggle competition — free Kaggle account, "Join Competition", then `kaggle competitions download -c aptos2019-blindness-detection` (needs Kaggle API token in `~/.kaggle/kaggle.json`) | ~3,662 labeled fundus images, single DR grade (0–4) per image. Largest labeled grading set. | M3 (grading, primary training set), M1 (variable image quality examples), M6 (validation split) |
| **IDRiD** | IEEE DataPort — free registration required, then download from the project page | Three sub-parts: (A) Disease Grading — DR grade + DME risk per image; (B) Segmentation — **pixel-level ground truth masks** for microaneurysms, hemorrhages, exudates, optic disc (only ~81 images, but pixel-accurate); (C) Localization — OD and fovea center coordinates | M2 (segmentation ground truth — most important dataset for this module), M3 (extra grading labels), M6 (benchmark numbers to compare against, since IDRiD is a common published benchmark) |
| **DRIVE** | grand-challenge.org — free registration, download from Data page | 40 retinal images with **manual vessel segmentation masks** (2 independent annotators on the test set) | M2 (vessel segmentation validation specifically) |
| **Messidor-2** | adcis.net — request access via their form (can take a day or two to be approved) | ~1,748 images; commonly used with the Krause et al. relabeled DR grades and has image-quality annotations | M1 (quality assessment ground truth), M3 (generalization/cross-dataset test — train on APTOS, test on Messidor-2 to show the model isn't overfit to one camera/population) |

## Practical notes
- **Start Kaggle/IEEE DataPort/grand-challenge registrations today** — IEEE
  DataPort and Messidor-2 approval can take time; don't let that block week 1.
- Store all datasets under one shared convention so paths don't break across
  everyone's machines, e.g.:
  ```
  data/
    aptos2019/train_images/, train.csv
    idrid/grading/, idrid/segmentation/, idrid/localization/
    drive/training/, drive/test/
    messidor2/
  ```
- Image sizes and formats differ a lot across these four datasets (JPEG vs
  TIFF, wildly different resolutions, different camera FOV crops) — this is
  actually useful, it's exactly the "variable image quality" problem the PS
  is asking you to handle. M1 should deliberately test against this variety
  rather than a single clean source.
- For **held-out validation** (M6), don't let the same images appear in both
  training and the final reported test metrics — hold out a fixed test split
  from APTOS + all of IDRiD-B (segmentation) + Messidor-2 up front, and don't
  touch it until final validation.

---

# PART 4 — PER-MODULE DETAIL

Full task brief for each module. (Shared background above already
covers what's common to all — this section is each module's specifics
only.)


# Context: Member 1 - Image Quality Assessment & Enhancement

## Your module: M1 - Quality Assessment & Enhancement

**Why this matters:** portable fundus cameras used in rural field conditions
produce images with focus blur, poor/uneven illumination, and incomplete
field of view. Feeding a bad image straight into grading produces
unreliable - and dangerously overconfident - results. Your module is the
gatekeeper: assess adequacy, fix what's fixable, reject what isn't with
clear feedback so the field operator can recapture.

## Function contract you must implement

```matlab
function result = assessAndEnhance(img)
% img: raw RGB fundus image, uint8, any resolution
%
% result.accepted              (logical)
% result.quality_score         (0-1 double)
% result.flags.focus_ok        (logical)
% result.flags.illumination_ok (logical)
% result.flags.fov_ok          (logical)
% result.enhanced_image        (uint8 RGB, same size - CLAHE + illumination
%                                normalized + denoised)
% result.feedback_msg          (string, human-readable, e.g. "Recapture:
%                                image too dark and off-center, ensure full
%                                retina is visible and increase illumination")
end
```

## What to build (in order)
1. **Focus/sharpness metric** - e.g. variance of Laplacian on the green
   channel (best vessel contrast); threshold empirically on sharp vs.
   deliberately-blurred images.
2. **Illumination check** - histogram of the luminance channel; flag if
   over/under-exposed regions exceed some percentage of the retinal area,
   or if mean brightness falls outside an acceptable band.
3. **Field-of-view check** - detect the circular retinal FOV mask (dark
   background vs. bright retinal disc); flag if the FOV circle is
   significantly cropped or off-center (full optic disc/fovea localization
   is Module 2's job, this is just a rough completeness check).
4. **Adaptive enhancement** for borderline (not outright rejected) images:
   - CLAHE (MATLAB's adapthisteq) per-channel or on luminance, tuned to
     avoid over-amplifying noise
   - Illumination normalization - subtract a heavily blurred version of the
     image (large-sigma Gaussian filter) to flatten uneven lighting, then
     re-add scaled base illumination
   - Denoising - non-local means or median filtering, applied carefully so
     it doesn't blur out microaneurysms (a few pixels wide), which Module 2
     depends on
5. **Rejection logic + feedback text** - combine the three flags into an
   accept/reject decision and a specific, actionable message.

## Testing
Use a deliberately mixed set: clean APTOS/IDRiD images, plus the same images
artificially degraded (blur, brightness/contrast shifts, synthetic
vignetting to fake a cropped FOV) to verify your metrics correctly separate
good from bad. Messidor-2 has real image-quality annotations if access is
approved in time - good for a final sanity check.

## Acceptance criteria
- Correctly rejects clearly bad images (heavy blur, very dark, major FOV
  crop) with a specific reason in feedback_msg.
- Enhancement visibly improves contrast/illumination on borderline images
  without destroying fine detail (spot check: are small vessels/lesions
  still visible after enhancement?).
- Runs on a single image in well under 1 second (Module 5's Simulink model
  needs your real per-image timing).

---

# Context: Member 2 - Retinal Structure Segmentation

## Your module: M2 - Retinal Structure Segmentation

**Why this matters:** DR severity grading is only trustworthy and
explainable if it's grounded in actual clinical lesions, not just a black-box
image classification. Your module extracts the structures ophthalmologists
actually look for, feeding both the grading model (Module 3) and the
explainability evidence (Module 4).

## Function contract you must implement

```matlab
function seg = segmentStructures(enhanced_image)
% enhanced_image: output of Module 1 (uint8 RGB)
%
% seg.vessel_mask               (logical, same size)
% seg.optic_disc.mask           (logical)
% seg.optic_disc.centroid       [x y]
% seg.optic_disc.radius         (double)
% seg.fovea_centroid            [x y]
% seg.microaneurysm_map.mask    (logical)
% seg.microaneurysm_map.candidates  (table: x, y, confidence, x_subpixel, y_subpixel)
% seg.exudate_mask              (logical)
% seg.hemorrhage_mask           (logical)
% seg.neovascularization_flag   (logical)
% seg.nv_regions                (logical mask)
% seg.lesion_features           (table - THIS FEEDS MODULE 3, freeze schema
%                                 with them early: e.g. ma_count, ma_total_area,
%                                 ex_count, ex_total_area, he_count, he_total_area,
%                                 vessel_density, nv_present)
end
```

## What to build (roughly in priority order - vessels and OD first, since
other structures are located relative to them)

1. **Vessel segmentation** - matched filter approach (Gaussian-profile
   kernels at multiple orientations) is a solid classical baseline; a small
   U-Net trained on DRIVE is a stronger option if time allows. DRIVE gives
   you ground truth to validate against directly.
2. **Optic disc localization** - the OD is the brightest, most saturated
   large circular region; combine intensity thresholding + `imfindcircles`
   (Hough transform) or region growing from the brightest connected
   component. IDRiD's localization set gives you ground-truth OD centers.
3. **Fovea localization** - typically ~2.5 optic disc diameters temporal to
   the OD, appears as a darker depression; use the OD position as a prior
   to constrain the search region rather than searching the whole image.
   IDRiD gives ground-truth fovea centers too.
4. **Microaneurysm detection (the hardest part)** - morphological top-hat
   filtering to enhance small round dark structures, candidate generation,
   then a simple classifier (shape/contrast features) to reject false
   positives from vessel crossings/noise. For "sub-pixel" localization,
   fit a 2D Gaussian (or use intensity-weighted centroid) around each
   candidate blob center rather than just taking the pixel centroid - this
   directly addresses the "sub-pixel microaneurysm detection" requirement
   in the problem statement. Validate against IDRiD-B's pixel-level MA masks.
5. **Exudate segmentation** - exudates are bright/yellowish; color-space
   thresholding (e.g. in a/b channels of Lab) combined with distance from
   OD (exudates near OD boundary need care not to be confused with OD
   itself) works as a baseline; IDRiD-B has ground truth here too.
6. **Hemorrhage detection/classification** - dark, irregular blotches
   (larger and less round than MAs); similar pipeline to MA detection but
   with different shape/size filters and classification into "hemorrhage"
   vs "microaneurysm" by size/shape.
7. **Neovascularization detection** - abnormal new vessel growth, typically
   near the OD or along major vessels; look for irregular, tortuous,
   fine-caliber vessel patterns not matching the normal vessel tree found
   in step 1. This is the hardest and rarest finding - a reasonable-effort
   heuristic (vessel density/tortuosity anomaly near OD) is acceptable for
   a prototype; be upfront in the report about its limitations.

## Testing
Validate directly against ground truth: DRIVE for vessels (compute
sensitivity/specificity of your vessel mask against the manual annotations),
IDRiD-B for MA/EX/HE/OD masks (Dice coefficient or per-lesion detection
sensitivity is a reasonable metric for the sparse lesion masks).

## Acceptance criteria
- Vessel segmentation Dice/sensitivity against DRIVE ground truth reported
  as a number, not just visual inspection.
- MA/EX/HE detection sensitivity against IDRiD-B ground truth reported.
- `lesion_features` table schema is agreed with Module 3's owner before
  either of you build very far into it - this is the one interface most
  likely to cause integration pain if left loose.

---

# Context: Member 3 - DR Severity Grading

## Your module: M3 - DR Severity Grading

**Why this matters:** this is the headline number the problem statement
scores you on - >90% sensitivity and >85% specificity for referable DR
(ICDR grade >=2). It also needs to be a *hybrid* model (image + lesion
features), not a pure black-box CNN, both because it's more accurate and
because Module 4 needs something more explainable than a plain CNN.

## Function contract you must implement

```matlab
function grading = gradeDR(enhanced_image, lesion_features)
% enhanced_image: output of Module 1 (uint8 RGB)
% lesion_features: table from Module 2 (schema frozen jointly with them)
%
% grading.grade                (int, 0-4, ICDR scale)
% grading.referable             (logical, grade >= 2)
% grading.class_probabilities   (1x5 double, sums to 1)
% grading.confidence            (0-1 double, CALIBRATED - see below, not
%                                 just the raw softmax max)
% grading.net                   (trained dlnetwork/SeriesNetwork - Module 4
%                                 needs this object directly for Grad-CAM)
end
```

## ICDR scale (for reference)
- 0: No DR
- 1: Mild NPDR (microaneurysms only)
- 2: Moderate NPDR
- 3: Severe NPDR
- 4: Proliferative DR (PDR)
Referable = grade >= 2 (this is the number the problem statement's
sensitivity/specificity target applies to, not raw 5-class accuracy).

## What to build
1. **CNN branch** - transfer learning from a pretrained network (ResNet-50
   or EfficientNet via MATLAB's pretrained network support), fine-tuned on
   APTOS (primary) + IDRiD grading labels. Freeze early layers, fine-tune
   later layers + a new classification head for 5 classes.
2. **Hand-crafted feature branch** - a simpler classifier (e.g. an ensemble
   of decision trees, or a small fully-connected network) trained directly
   on Module 2's `lesion_features` table (MA/EX/HE counts and areas, vessel
   density, NV flag).
3. **Hybrid fusion** - combine the two branches, either by concatenating the
   CNN's penultimate-layer features with the lesion features before a final
   classifier, or by a simple weighted/learned ensemble of the two branches'
   probabilities. Keep both branches trainable/testable independently too -
   Module 6 needs to compare "CNN-only" vs "features-only" vs "hybrid" as
   three separate benchmark numbers.
4. **Calibration** - raw softmax confidence tends to be overconfident. Apply
   temperature scaling or Platt scaling on a held-out validation set so that
   `confidence` actually reflects real-world reliability (a 0.9-confidence
   prediction should be right ~90% of the time). This calibrated confidence
   is what tells the ophthalmologist which cases most need their attention.

## Class imbalance warning
DR datasets are heavily skewed toward "no DR" (grade 0). Use class weighting
or resampling during training, and report sensitivity/specificity per class
(especially referable vs non-referable), not just overall accuracy - overall
accuracy can look great while completely missing referable cases.

## Testing / validation plan
- Train on APTOS + IDRiD-A, hold out a fixed test split (agree this split
  with Module 6, don't touch it until final validation).
- Report sensitivity/specificity/AUC for referable DR (grade >=2) as the
  headline number.
- Test generalization on Messidor-2 if access comes through in time (train
  on APTOS, test on Messidor-2, to show the model isn't overfit to one
  camera/population - a real deployment concern for rural India where
  camera hardware varies).

## Acceptance criteria
- Referable-DR sensitivity > 90%, specificity > 85% on your held-out test
  split (this is the problem statement's explicit target).
- Confidence scores are demonstrably calibrated (e.g. a reliability
  diagram/calibration curve, not just claimed).
- `net` object is directly usable by Module 4's `gradCAM` call without
  extra conversion.

---

# Context: Member 4 - Explainability Module

## Your module: M4 - Explainability

**Why this matters:** this is what turns a black-box AI into a clinically
usable tool. The problem statement explicitly requires the ophthalmologist
to be able to validate a case in under 30 seconds - your output IS the
interface they'll actually look at, so clarity matters as much as
correctness.

## Function contract you must implement

```matlab
function explanation = explainGrading(enhanced_image, net, seg, grading)
% enhanced_image: output of Module 1
% net: trained network object from Module 3's grading.net
% seg: full segmentation struct from Module 2
% grading: full grading struct from Module 3
%
% explanation.cam_heatmap      (double, 0-1 normalized, same size as image)
% explanation.overlay_image    (uint8, heatmap blended onto enhanced_image)
% explanation.lesion_evidence  (table: lesion_type, location, confidence,
%                                cam_overlap_pct - i.e. which detected
%                                lesions the CAM heatmap actually agrees are
%                                important)
% explanation.report_file      (path to a generated annotated PDF/PNG report)
end
```

## What to build
1. **Grad-CAM** - use MATLAB's built-in `gradCAM` function on Module 3's
   trained network to get a class-activation heatmap for the predicted
   grade. Overlay it on the enhanced image (semi-transparent, using a
   perceptually distinct colormap like 'jet' or 'hot').
2. **Lesion-level evidence correlation** - this is the key differentiator
   from a plain Grad-CAM output. For each lesion Module 2 detected
   (microaneurysm, exudate, hemorrhage, NV region), compute what fraction of
   that lesion's pixels fall within the high-activation region of the CAM
   heatmap. This produces a structured table like: "3 microaneurysms
   detected, 2 strongly correlated with model attention (>70% overlap), 1
   weakly correlated" - this is what actually lets an ophthalmologist agree
   or disagree with the model's reasoning in seconds, rather than squinting
   at a heatmap and guessing.
3. **Calibrated confidence display** - surface Module 3's calibrated
   confidence score prominently; flag low-confidence or high-disagreement
   (CAM vs detected lesions don't line up) cases as needing extra scrutiny.
4. **Automated annotated report** - a single-page PDF or image combining:
   original image, CAM overlay, detected lesion markers, predicted grade +
   confidence, and the lesion evidence table. MATLAB Report Generator
   (`mlreportgen.dom`/`mlreportgen.report`) is one way to do this; a
   composed image via `insertText`/`insertMarker`/`imtile` plus `imwrite`
   as a simpler alternative if Report Generator access is limited.

## Testing
Since "clinically useful" is inherently subjective, the best test available
to a hackathon team is: (1) internal consistency checks - does the CAM
heatmap actually concentrate on the image regions Module 2 flagged as
lesions, for cases with an obviously correct grade; (2) if anyone on the
team has any clinical/medical contact (even a friend/relative in med
school), a quick informal "can you tell what this is flagging within 30
seconds" review is valuable evidence for the pitch.

## Acceptance criteria
- Grad-CAM heatmap visibly concentrates over actual retinal
  lesions/abnormalities for correctly-graded test cases, not scattered
  randomly or centered on image borders/artifacts.
- Lesion evidence table is generated automatically with no manual steps.
- A full annotated report renders in well under 30 seconds of *review* time
  (i.e. it's readable at a glance, not requiring the reviewer to scroll or
  decode a wall of text).

---

# Context: Member 5 - Simulink Workflow Simulation

## Your module: M5 - Simulink Workflow Simulation

**Why this matters:** an accurate AI model is useless if the deployment
can't handle real patient volume. This module answers the operational
question: for a district serving 100,000+ patients/year, how many cameras,
servers, and reviewing ophthalmologists are actually needed, and where does
the system bottleneck? This is a *systems* problem, largely independent of
how good Modules 1-4's algorithms are - you can start with estimated timing
numbers and refine them once real modules exist.

## What you're modeling (not a MATLAB function contract, but a Simulink
model + driver script `runScreeningSim.m`)

Model the pipeline as a queueing/throughput system with these stages:
1. **Image acquisition** - patients arrive at a rate (derive from 100,000
   patients/year -> ~275/day average across the district, but plan for
   peak-hour bursts, not just the average, since clinics don't get a steady
   drip of patients all day).
2. **Transmission/bandwidth** - image file size (MB) divided by available
   network bandwidth (Mbps) at rural centers gives an upload/transmission
   delay per image; rural bandwidth is often poor and variable - model this
   as a distribution, not a fixed number, if you want to be more realistic.
3. **Processing throughput** - per-image processing time for Modules 1-4
   combined (get REAL numbers from those team members once their code
   exists; use a reasonable placeholder like 2-5 seconds/image until then).
   Model this as one or more parallel processing servers.
4. **Ophthalmologist review capacity** - only flagged/referable/low-confidence
   cases need human review (~30 seconds each per the problem statement);
   non-referable high-confidence cases can be auto-cleared. Model the
   reviewing ophthalmologist(s) as a server with a queue, and use Module
   3's expected referable-case rate (roughly, DR prevalence x referable
   fraction) to estimate what fraction of images actually need review.

## Suggested approach
- If your MATLAB license includes **SimEvents**, use its queue/server/delay
  blocks directly for a proper discrete-event simulation - this maps
  naturally onto the stages above.
- If SimEvents isn't available, a Simulink model using standard blocks
  (or even a MATLAB script implementing M/M/1 or M/M/c queueing formulas)
  is an acceptable substitute - the deliverable is the *analysis and
  recommendation*, the specific tool is secondary.
- Parameterize everything (arrival rate, bandwidth, processing time per
  stage, number of servers/ophthalmologists) so you can run multiple
  scenarios and produce a chart of, e.g., "average wait time vs. number of
  reviewing ophthalmologists" to justify a specific staffing recommendation.

## What to produce
- A `.slx` Simulink model file.
- A driver script that runs multiple scenarios (varying number of centers,
  servers, ophthalmologists) and outputs throughput/wait-time results.
- A clear final recommendation: e.g. "for a district of 100,000 patients/year,
  X camera units, Y processing servers, and Z reviewing ophthalmologists
  achieve <2 hour turnaround with no more than W% queue overflow risk,"
  plus identification of which stage is the bottleneck (acquisition,
  bandwidth, processing, or review).

## Acceptance criteria
- Model runs and produces throughput numbers, not just a diagram.
- At least one chart showing a resource-allocation tradeoff (e.g. wait time
  vs. number of ophthalmologists).
- A specific, numeric staffing/resource recommendation for the 100,000-
  patient/year target, with the bottleneck stage clearly identified.

---

# Context: Member 6 - Integration, Validation & Demo (Team Lead role)

## Your module: M6 - Integration, Validation, Benchmarking, Demo/PPT

**Why this matters:** SIH judges score the *integrated, working prototype*
and the *evidence it actually works*, not five disconnected scripts. You own
making sure Modules 1-5 actually plug together, that the headline numbers
(>90% sensitivity, >85% specificity for referable DR, "outperforms any
single-technique approach") are real and reproducible, and that the demo/PPT
tells that story clearly.

## What to build

1. **Pipeline driver** (`runPipeline.m` or a MATLAB App Designer app) that:
   - Takes a folder of raw fundus images
   - Calls M1 -> M2 -> M3 -> M4 in sequence for each image
   - Logs results to a table (per image: accepted?, grade, referable?,
     confidence, path to report)
   - Handles the reject case gracefully (image doesn't proceed past M1,
     feedback is logged/shown)
   Build this in Week 1 using **stub versions** of M1-M4 that return
   dummy-but-correctly-shaped structs (matching the contracts in
   `00_ARCHITECTURE.md`) so the pipeline runs end-to-end from day one, then
   swap in real modules as teammates finish them.

2. **Validation / benchmarking**, this is the evidence for the problem
   statement's specific claims:
   - Fix a held-out test split (agree with Module 3's owner, don't touch
     it until final validation) across APTOS + IDRiD + (Messidor-2 if
     access comes through).
   - Compute and report: sensitivity, specificity, AUC for referable DR
     (grade >=2) on the full integrated pipeline.
   - Run the three-way comparison described in `00_ARCHITECTURE.md` section
     5: CNN-only vs. hand-crafted-features-only vs. integrated hybrid
     pipeline, to substantiate "the integrated pipeline outperforms any
     single-technique approach."
   - Compare your numbers against published results on the same datasets
     (search for published APTOS/IDRiD DR-grading benchmark papers) so you
     can honestly say how you compare - even if you don't beat the SOTA,
     being close while being explainable (which most SOTA black-box models
     aren't) is itself a legitimate pitch.

3. **Demo** - a MATLAB App Designer GUI or a narrated Live Script that
   takes a sample image through the full pipeline live: quality check ->
   segmentation overlay -> grade + confidence -> Grad-CAM + lesion evidence
   report -> (mention, don't need to run live) Simulink resource projection.
   This is what you'll actually show judges - prioritize it being reliable
   and fast over being feature-complete.

4. **Final report/PPT** - pull together: the clinical problem and why
   existing approaches fall short (black-box, unvalidated, fail on variable
   field image quality - this is literally the problem statement's framing,
   use it), your architecture, your validation numbers, the Simulink
   deployment-scale story, and a short live or recorded demo clip as backup
   in case live demo has issues on presentation day.

## Coordination responsibilities
- Keep `00_ARCHITECTURE.md`'s interface contracts up to date if the team
  agrees to change anything mid-project, and make sure everyone knows.
- Chase down dataset access approvals (IEEE DataPort, Messidor-2) early -
  these can take days, don't let them block week 1 for the whole team.
- Run the full pipeline end-to-end at least every few days, not just at the
  very end, so integration bugs surface while there's still time to fix them.

## Acceptance criteria
- Pipeline runs end-to-end on a real image with zero manual intervention.
- Validation numbers are real (computed from an actual test run you can
  reproduce), with the three-way single-technique-vs-integrated comparison
  clearly documented.
- Demo runs reliably without live-coding risk on presentation day.

---
