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
