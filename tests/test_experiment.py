import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import optuna
import pytest
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader

import bci_experiment as experiment

torch.set_num_threads(1)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TinyNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, **unused):
        super().__init__()
        self.linear = nn.Linear(n_channels * n_times, n_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def synthetic_subject(subject):
    rng = np.random.default_rng(subject)
    return rng.normal(size=(8, 2, 64)), np.array([0, 1] * 4)


def test_all_nested_folds_are_subject_disjoint():
    folds = experiment.subject_folds([1, 2, 3, 4])
    assert len(folds) == 4
    for fold in folds:
        assert fold.test_subject not in fold.train_subjects
        assert len(fold.inner_folds) == 3
        for train, valid in fold.inner_folds:
            assert valid not in train
            assert fold.test_subject not in (*train, valid)
            assert set((*train, valid)) == set(fold.train_subjects)


@pytest.mark.parametrize("subjects", [[], [1, 2], [1, 1, 2], [0, 1, 2], [True, 2, 3], [1.5, 2, 3]])
def test_invalid_subject_protocol_is_rejected(subjects):
    with pytest.raises(ValueError):
        experiment.subject_folds(subjects)


def test_outer_subject_is_not_loaded_until_after_final_fit(monkeypatch):
    events = []
    encoder = LabelEncoder().fit([0, 1])

    def load(subject):
        events.append(("load", subject))
        return np.full((4, 2, 64), subject, dtype=float), np.array([0, 1, 0, 1])

    def tune(data, folds, **kwargs):
        outer = next(iter({1, 2, 3} - data.keys()))
        assert all(outer not in (*train, valid) for train, valid in folds)
        events.append(("tune", outer))
        return {}, 0.5

    def fit(data, subjects, params, **kwargs):
        events.append(("fit", next(iter({1, 2, 3} - set(subjects)))))
        return object(), encoder

    monkeypatch.setattr(experiment, "tune_subjects", tune)
    monkeypatch.setattr(experiment, "fit_model", fit)
    monkeypatch.setattr(experiment, "evaluate_accuracy", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(experiment, "csp_lda_fit_predict", lambda X, y, test: np.zeros(len(test)))
    rows = experiment.run_loso([1, 2, 3], load)
    for offset, row in zip(range(0, len(events), 5), rows):
        outer = row["test_subject"]
        chunk = events[offset:offset + 5]
        assert chunk[2:] == [("tune", outer), ("fit", outer), ("load", outer)]
        assert outer not in [subject for _, subject in chunk[:2]]
        assert row["test_samples"] == 4


def test_changing_outer_labels_does_not_change_selected_parameters(monkeypatch):
    monkeypatch.setattr(experiment, "csp_lda_fit_predict", lambda X, y, test: np.zeros(len(test)))
    def changed(subject):
        X, y = synthetic_subject(subject)
        return X, 1 - y if subject == 1 else y
    first = experiment.run_loso([1, 2, 3], synthetic_subject, trials=2, epochs=1, model_factory=TinyNet)
    second = experiment.run_loso([1, 2, 3], changed, trials=2, epochs=1, model_factory=TinyNet)
    assert first[0]["best_params"] == second[0]["best_params"]
    assert first[0]["inner_accuracy"] == second[0]["inner_accuracy"]


def test_seeded_sampler_and_training_reproduce_on_cpu():
    data = {s: synthetic_subject(s) for s in (2, 3)}
    folds = (((2,), 3), ((3,), 2))
    options = dict(trials=2, epochs=1, seed=17, model_factory=TinyNet)
    assert experiment.tune_subjects(data, folds, **options) == experiment.tune_subjects(data, folds, **options)


def test_real_eegnet_train_and_evaluate_every_sample():
    X, y = synthetic_subject(1)
    experiment.set_seed(2)
    model = experiment.EEGNet(2, 64, F1=4, D=1, dropout=0)
    before = model.head[-1].weight.detach().clone()
    loader = DataLoader(experiment.EEGDataset(X, y), batch_size=3)
    trained = experiment.train_model(model, loader, epochs=1, lr=0.001, device="cpu")
    assert trained is model
    assert not torch.equal(before, model.head[-1].weight)
    accuracy = experiment.evaluate_accuracy(model, X, y, LabelEncoder().fit(y), batch_size=3)
    assert 0 <= accuracy <= 1


def test_device_failure_is_not_silently_retried(monkeypatch):
    model = TinyNet(2, 64, 2)
    calls = []
    def fail(device):
        calls.append(device)
        raise RuntimeError("synthetic device failure")
    monkeypatch.setattr(model, "to", fail)
    with pytest.raises(RuntimeError, match="synthetic device failure"):
        experiment.train_model(model, [], epochs=1, lr=0.001)
    assert calls == [torch.device("cpu")]


def test_unknown_heldout_class_is_not_silently_dropped():
    X, y = synthetic_subject(1)
    y[0] = 9
    with pytest.raises(ValueError):
        experiment.evaluate_accuracy(TinyNet(2, 64, 2), X, y, LabelEncoder().fit([0, 1]))


def test_empty_training_loader_fails():
    with pytest.raises(ValueError, match="empty"):
        experiment.train_model(TinyNet(2, 64, 2), [], epochs=1, lr=0.001)


@pytest.mark.parametrize("epochs,lr", [(0, 0.01), (1, 0), (1, float("nan"))])
def test_invalid_training_budget_fails(epochs, lr):
    with pytest.raises(ValueError):
        experiment.train_model(TinyNet(2, 64, 2), [], epochs=epochs, lr=lr)


def test_constant_epoch_normalization_is_finite():
    X = np.ones((4, 2, 64))
    assert np.array_equal(experiment.zscore_epochs(X), np.zeros_like(X))


@pytest.mark.parametrize("X,y", [(np.ones((2, 64)), [0, 1]), (np.ones((2, 2, 64)), [0]),
                                (np.full((2, 2, 64), np.nan), [0, 1]), (np.ones((2, 2, 64)), [0., 1.])])
def test_invalid_epochs_or_labels_fail(X, y):
    with pytest.raises(ValueError):
        experiment.EEGDataset(X, y)


def test_real_csp_lda_small_synthetic_dataset():
    X, y = synthetic_subject(2)
    prediction = experiment.csp_lda_fit_predict(X, y, synthetic_subject(3)[0])
    assert prediction.shape == y.shape
    assert set(prediction).issubset(set(y))


def test_notebook_is_clean_valid_and_uses_shared_protocol():
    path = Path(__file__).resolve().parents[1] / "ml_bci_res.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert set(notebook.metadata) == {"kernelspec", "language_info"}
    source = []
    for cell in notebook.cells:
        assert not cell.metadata and "attachments" not in cell
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            ast.parse(cell.source)
            source.append(cell.source)
    joined = "\n".join(source)
    assert "run_loso(" in joined and "allow_download = False" in joined
    assert "def train_model" not in joined and "def objective" not in joined


def test_notebook_download_is_disabled_before_dataset_loader(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "ml_bci_res.ipynb"
    doc = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"cfg": type("Config", (), {"allow_download": False})()}
    exec("".join(doc["cells"][4]["source"]), namespace)
    with pytest.raises(RuntimeError, match="allow_download"):
        namespace["load_preprocess"]()
