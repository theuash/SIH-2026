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
