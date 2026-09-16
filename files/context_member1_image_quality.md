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
