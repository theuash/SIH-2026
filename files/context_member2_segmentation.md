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
with their own AI assistant, and wired together at the end. **You are
building ONE module.** Other modules will call your function using the exact
input/output contract below - do not change the contract without telling the
team, since other people depend on it.

Tools available: MATLAB Image Processing Toolbox, Computer Vision Toolbox,
Deep Learning Toolbox, Medical Imaging Toolbox, Simulink, Statistics and
Machine Learning Toolbox.

Target metrics for the *overall* system (not just your module): >90%
sensitivity and >85% specificity for referable DR (grade >=2).
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
