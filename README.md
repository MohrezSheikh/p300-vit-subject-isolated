# P300-ViT Leak-Free — xDAWN → CWT → Vision Transformer under subject-isolated evaluation

Standalone evaluation repository for the manuscript *"Leak-Free Evaluation of
xDAWN-Enhanced P300 Features with a Vision Transformer for Schizophrenia
Screening"* (companion to a benchmark paper on generalization barriers in
clinical EEG; no code dependency between the two).

The pipeline is the originally proposed one — xDAWN spatial filtering of
stimulus-locked auditory-oddball EEG, continuous-wavelet scalograms, and a
Vision Transformer classifier — evaluated under a **frozen, pre-registered,
leak-free protocol**, and contrasted on identical data with the literature's
trial-randomized (leaky) protocol.

## What it computes

| Stage | Output | Manuscript table/figure |
|---|---|---|
| `participant` | strict subject-wise 5-fold CV (clinical N=40, ASZED, pooled), pooled OOF subject metrics + per-fold trial probabilities | participant-level results |
| `trialwise` | the same variant under the leaky trial-randomized protocol | window-level vs participant-level contrast |
| `selective` | reject-option coverage–accuracy curves (θ sweep) from saved OOF probabilities | selective-classification table |
| `fam` | input-gradient attribution + pre/post-stimulus mass (stimulus locking) | interpretability figure |

## Run

Everything runs as the Kaggle kernel `p300vit-standalone` (GPU T4); the code
ships inside the notebook as an embedded blob, so the kernel needs only the
data attachments:

```
kaggle kernels push -p kaggle/          # after scripts/embed_code.py
kaggle kernels status mmdation/p300vit-standalone
```

Local synthetic checks (no data required):

```
python tests/test_leakfreedom.py
```

## Layout

```
src/p300vit/          config, data adapters (.cnt + ASZED EDF), preprocessing,
                      xDAWN (training-fold-only fitting), CWT-ViT model,
                      training engine, subject-wise CV harness, trial-randomized
                      protocol, statistics (bootstrap CIs, McNemar, calibration)
scripts/run_all.py    the four-stage driver (resumable per stage/experiment)
kaggle/               kernel notebook + metadata
docs/PROTOCOL.md      the frozen pre-registered protocol
tests/                synthetic leak-freedom checks
```

## Leakage contract

`fit_xdawn` receives training-fold subject epochs only; the CV harness enforces
it and `tests/test_leakfreedom.py` verifies that filters differ between folds.
Thresholds are selected on inner-validation subjects and applied unchanged to
held-out test folds.

## License

MIT — see [LICENSE](LICENSE).
