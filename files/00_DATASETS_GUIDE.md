# Dataset Guide — PS 26038

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
