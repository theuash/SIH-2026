# Team Plan — 6 Members, SIH 2026 PS 26038

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
