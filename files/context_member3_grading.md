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
