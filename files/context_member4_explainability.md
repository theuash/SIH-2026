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
