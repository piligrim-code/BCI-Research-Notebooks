"""Synthetic execution check, not EEG accuracy evidence."""
import json
import numpy as np
import optuna
import torch
from bci_experiment import run_loso


def main():
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def load_subject(subject):
        rng = np.random.default_rng(subject)
        return rng.normal(size=(8, 2, 64)), np.array([0, 1] * 4)
    results = run_loso([1, 2, 3], load_subject, trials=1, epochs=1)
    print(json.dumps({"scope": "synthetic execution check; not an EEG benchmark",
                      "folds": [{"test_subject": r["test_subject"], "train_subjects": r["train_subjects"],
                                 "test_samples": r["test_samples"]} for r in results]}))


if __name__ == "__main__":
    main()
