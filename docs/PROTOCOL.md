# Pre-Registered Evaluation Protocol — xDAWN→CWT→ViT (v8 companion paper)

**Status:** frozen before the definitive runs. Any deviation must be documented in the
manuscript's limitations, never silently. Runs execute inside the
`mmdation/p300vit-standalone` Kaggle kernel; nothing runs against local data.

## Objective
Evaluate the old paper's pipeline (xDAWN spatial filtering → continuous-wavelet
scalograms → Vision Transformer) under leak-free, subject-isolated evaluation, and
quantify the evaluation-level effect on identical data and an identical model.

## Model
- Variant `xdawn_cwt_vit`: CWTViT with `n_channels = cfg.xdawn.n_components (K = 4)`,
  Morlet front-end (1–30 Hz, fixed kernels), patch 16, embed 256, depth 6, heads 8,
  dropout 0.1 — identical to the benchmark's CWT 2D-ViT except for the xDAWN-projected
  4-channel input.
- Spatial filters: `fit_xdawn` on **training-fold subjects only**, `n_components = 4`,
  Ledoit-Wolf-style trace shrinkage per `XDAWNConfig`; applied unchanged to validation
  and test folds (`xdawn_mode = "xdawn"` in `run_cv`).

## Data
| Cohort | Loader | N | Channels |
|---|---|---|---|
| Clinical (Sabeti) | `local` via `mmdation/p300-ieee` | as recorded | 20 |
| ASZED | `aszed` via `mmdation/aszedeeg` | as recorded | 19 |
| Pooled | `pooled`, folds stratified by site × label | both | unified |

Preprocessing is the benchmark's frozen pipeline (resample 200 Hz, notch, 0.5–45 Hz
zero-phase band-pass, CAR, epochs −200…800 ms, baseline −200…0 ms, per-trial z-scoring).

## Partitions and calibration (frozen)
1. **Participant-level:** subject-wise StratifiedKFold, 5 folds, `seed = cfg.cv.seed`;
   inner stratified split of training subjects for early stopping and the Youden
   threshold; thresholds applied unchanged to held-out test folds. Reported: pooled OOF
   subject AUC, accuracy/sensitivity/specificity at the 0.5 threshold, fold-mean ± SD AUC,
   bootstrap CIs (from the audit's statistics stack).
2. **Window-level (leaky contrast):** `trialwise_cv` (event-shuffled folds; xDAWN still
   fitted in-fold) on `local` and `aszed` — the literature's protocol, reported for the
   same variant.
3. **Selective classification:** abstain when `max(p, 1−p) < θ`, θ ∈ {0.50, 0.55, …, 0.95},
   computed from the saved participant-level OOF subject probabilities. Report coverage,
   accuracy, sensitivity, specificity per θ. No threshold re-fitting.
4. **FAM / stimulus locking:** input-gradient attribution of the true-class score through
   the frozen wavelet front-end on the **last participant-level fold's held-out trials**
   (checkpoint `fold_4/model.pt`); report pre- vs post-stimulus attribution mass and peak
   latency. Success criterion: post-stimulus mass ≥ 80%.

## What counts as the result
Only artifacts written by the kernel
(`results/xdawn_vit/**/summary.json`, `oof_subjects.npz`, `oof_trials.npz`,
`selective_classification.json`, `fam_summary.json`, and the consolidated
`xdawn_vit_summary.json`) enter the manuscript. Expected outcome bands are recorded in
the analysis log **before** the run: participant-level accuracy 55–75%, pooled OOF AUC
0.55–0.75; window-level accuracy 90–99%. Any outcome inside these bands is reported as-is.

## Forbidden after the run
Changing folds, seeds, thresholds, model hyperparameters, or cohort definitions after
seeing results; reporting a re-run under a different protocol as the primary result;
reporting window-level numbers in the abstract.
