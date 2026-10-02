"""Subject-isolated EEG experiment. Importing this module never downloads data."""
from dataclasses import dataclass
import random

import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_epochs(X, y):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 3 or not all(X.shape) or y.ndim != 1 or len(y) != len(X):
        raise ValueError("Expected nonempty aligned epochs [samples, channels, times] and labels")
    if not np.isfinite(X).all():
        raise ValueError("Epochs must be finite")
    return X, y


def zscore_epochs(X):
    X = np.asarray(X, dtype=np.float32)
    return (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)


class EEGDataset(Dataset):
    def __init__(self, X, y):
        X, y = validate_epochs(X, y)
        if not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
            raise ValueError("Use nonnegative encoded integer labels")
        self.X = torch.from_numpy(zscore_epochs(X)).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class EEGNet(nn.Module):
    """The notebook's compact EEGNet-style architecture, not a reference reproduction."""
    def __init__(self, n_channels, n_times, n_classes=2, F1=16, D=2, dropout=0.5):
        super().__init__()
        if n_channels < 1 or n_times < 32 or n_classes < 2:
            raise ValueError("Need channels >= 1, times >= 32 and classes >= 2")
        self.n_classes = n_classes
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 51), padding=(0, 25), bias=False), nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(F1 * D, n_classes))

    def forward(self, x):
        return self.head(self.separable(self.depthwise(self.firstconv(x))))


def train_model(model, train_loader, *, epochs, lr, device="cpu"):
    """Fixed-epoch training; no validation/test data and no implicit GPU fallback."""
    if not isinstance(epochs, int) or epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Need positive integer epochs and finite positive learning rate")
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            seen += len(yb)
        if not seen:
            raise ValueError("Training loader is empty")
    return model


def evaluate_accuracy(model, X, y, encoder, *, device="cpu", batch_size=64):
    # Unknown held-out classes are an invalid protocol, not rows to discard.
    encoded = encoder.transform(y)
    loader = DataLoader(EEGDataset(X, encoded), batch_size=batch_size, shuffle=False, drop_last=False)
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


@dataclass(frozen=True)
class SubjectFold:
    test_subject: int
    train_subjects: tuple
    inner_folds: tuple


def subject_folds(subjects):
    ids = tuple(subjects)
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("Nested subject evaluation requires at least three distinct subjects")
    if any(not isinstance(s, (int, np.integer)) or isinstance(s, bool) or s < 1 for s in ids):
        raise ValueError("Subject IDs must be positive integers")
    folds = []
    for outer in ids:
        train = tuple(s for s in ids if s != outer)
        inner = tuple((tuple(s for s in train if s != valid), valid) for valid in train)
        folds.append(SubjectFold(outer, train, inner))
    return tuple(folds)


def combine_subjects(data, subjects):
    arrays = [validate_epochs(*data[s]) for s in subjects]
    return np.concatenate([a[0] for a in arrays]), np.concatenate([a[1] for a in arrays])


def fit_model(data, subjects, params, *, epochs, seed, device, model_factory=EEGNet):
    X, y = combine_subjects(data, subjects)
    encoder = LabelEncoder().fit(y)
    if len(encoder.classes_) < 2:
        raise ValueError("Every training split must contain at least two classes")
    set_seed(seed)
    loader = DataLoader(EEGDataset(X, encoder.transform(y)), batch_size=params["batch_size"],
                        shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(seed))
    model = model_factory(n_channels=X.shape[1], n_times=X.shape[2], n_classes=len(encoder.classes_),
                          F1=params["F1"], D=params["D"], dropout=params["dropout"])
    return train_model(model, loader, epochs=epochs, lr=params["lr"], device=device), encoder


def tune_subjects(data, inner_folds, *, trials, epochs, seed, device="cpu", model_factory=EEGNet):
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("At least one tuning trial is required")

    def objective(trial):
        params = {
            "F1": trial.suggest_categorical("F1", [8, 16, 32]),
            "D": trial.suggest_int("D", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.7),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        scores = []
        for index, (train_ids, valid_id) in enumerate(inner_folds):
            model, encoder = fit_model(data, train_ids, params, epochs=epochs, seed=seed + index,
                                       device=device, model_factory=model_factory)
            scores.append(evaluate_accuracy(model, *data[valid_id], encoder, device=device))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return dict(study.best_params), float(study.best_value)


def csp_lda_fit_predict(X_train, y_train, X_test, n_components=6):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    csp = CSP(n_components=min(n_components, X_train.shape[1]), reg=None, log=True, norm_trace=False)
    Xtr = csp.fit_transform(np.asarray(X_train, dtype=np.float64), y_train)
    clf = LinearDiscriminantAnalysis().fit(Xtr, y_train)
    return clf.predict(csp.transform(np.asarray(X_test, dtype=np.float64)))


def run_loso(subjects, load_subject, *, trials=3, epochs=6, seed=42, device="cpu", model_factory=EEGNet):
    """Inner LOSO selects parameters. Outer data is loaded only after final fitting."""
    results = []
    for index, fold in enumerate(subject_folds(subjects)):
        fold_seed = seed + index * 1000
        data = {s: validate_epochs(*load_subject(s)) for s in fold.train_subjects}
        best, inner_score = tune_subjects(data, fold.inner_folds, trials=trials, epochs=epochs,
                                         seed=fold_seed, device=device, model_factory=model_factory)
        model, encoder = fit_model(data, fold.train_subjects, best, epochs=epochs, seed=fold_seed,
                                   device=device, model_factory=model_factory)
        X_test, y_test = validate_epochs(*load_subject(fold.test_subject))
        nn_score = evaluate_accuracy(model, X_test, y_test, encoder, device=device)
        X_train, y_train = combine_subjects(data, fold.train_subjects)
        pred_csp = csp_lda_fit_predict(X_train, encoder.transform(y_train), X_test)
        results.append({
            "test_subject": fold.test_subject,
            "acc_csp": float(np.mean(pred_csp == encoder.transform(y_test))),
            "acc_eegnet": nn_score,
            "inner_accuracy": inner_score,
            "best_params": best,
            "train_subjects": list(fold.train_subjects),
            "test_samples": len(y_test),
            "seed": fold_seed,
        })
    return results
