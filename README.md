# EEG Motor-Imagery Research Notebook

A small EEGNet-style versus CSP/LDA experiment with subject-isolated model
selection. This repository is a research example, not a validated decoder or
clinical system. No new real-EEG accuracy results are included in this revision.

## What Was Corrected

The original notebook optimized neural hyperparameters on the outer held-out
subject and then reported a score on that same subject. Those saved scores
have been removed; they are not independent held-out evidence.

The shared `bci_experiment.py` now implements:

1. Outer leave-one-subject-out evaluation with at least three distinct subjects.
2. Inner leave-one-subject-out folds using only the outer training subjects.
   Optuna maximizes the unweighted mean inner-subject accuracy.
3. Fresh final training on all outer training subjects using the selected
   parameters and a predeclared fixed epoch budget.
4. Loading/scoring the outer test subject after final neural training. It does
   not select parameters, epochs or checkpoints.

Label encoders are fitted on each training split. Unseen held-out classes cause
an explicit error rather than silently removing examples. Epoch normalization
is per example. CSP/LDA fits only the outer training data; its component count
is fixed to at most six channels, not tuned on held-out scores.

The network preserves the notebook's architecture. Its layer named
`separable` is an ordinary convolution; this is an EEGNet-style example, not
a claim of exact reproduction of the published EEGNet architecture.

## Install And Test

Use a dedicated Python 3.12 environment. For CPU-only testing:

```sh
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python demo.py
python -m pip check
```

The demo runs tiny synthetic epochs through actual neural training and CSP/LDA
and reports split/sample counts, not a fabricated EEG quality metric. Tests
exercise outer-label independence, fold disjointness, seeded CPU repeatability,
actual model updates, complete scoring, invalid inputs and notebook structure.
Importing the module or running the synthetic suite does not acquire EEG data.

Dependencies are bounded ranges, not a fully locked scientific environment.
CI checks clean CPU environments on Windows and Linux. GPU/MPS numerical
behavior has not been qualified by those checks.

## Real Dataset Run

Open `ml_bci_res.ipynb` with the repository root as the working directory.
The notebook uses MNE's EEGBCI loader for motor-imagery runs 4, 8 and 12.
It starts with `allow_download = False` and a repository-local ignored data
directory. Review the upstream dataset's provenance/terms and expected download
size before explicitly setting `cfg.allow_download = True`.

Choose subjects, filtering, trial window, epochs and tuning budget before the
experiment. The default three subjects/three trials are a small demonstration,
not a population-level benchmark. Repeatedly changing these choices after
looking at outer scores would compromise their interpretation.

The device is explicit and defaults to CPU. A failed accelerator operation
propagates; there is no silent CPU retry on a partially trained model. To retry
on CPU, rerun the experiment from fresh initialization with `cfg.device="cpu"`.
Seeds cover the sampler, model initialization and shuffle generator, but do not
promise bitwise equivalence across library versions or hardware.

For meaningful results record the dataset/version, subjects, preprocessing,
seeds, dependency versions, hardware, complete per-subject sample counts and
scores. Recompute results after this protocol correction; no old figure is a
replacement for that run. Keep acquired data and generated outputs untracked.

The existing MIT code license is unchanged. Dataset rights are separate.
