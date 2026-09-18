"""Train SEED EEG emotion classifiers from precomputed DE features.

Improvements over baseline:
- Features   : de_LDS + dasm_LDS + rasm_LDS (580 dims instead of 310)
- Architecture: Residual MLP blocks — skip connections prevent gradient vanishing
- Augmentation: Mixup — interpolates EEG windows for smoother decision boundaries
- Ensemble   : soft-vote MLP softmax + calibrated SVM confidence

Reference:
    Zheng & Lu, "Investigating Critical Frequency Bands and Channels for
    EEG-based Emotion Recognition with Deep Neural Networks", IEEE TAMD, 2015.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from sklearn.calibration import CalibratedClassifierCV
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# Features to concatenate: de(310) + dasm(135) + rasm(135) = 580 dims
FEATURE_KEYS = ["de_LDS", "dasm_LDS", "rasm_LDS"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_seed_features(
    feature_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load SEED DE + asymmetry features from official .mat archives.

    Concatenates de_LDS (62ch×5band=310), dasm_LDS (27ch×5=135),
    rasm_LDS (27ch×5=135) → 580-dim feature vector per 1-second window.

    Returns
    -------
    features : (N, 580) float32
    labels   : (N,)     int64   — 0=Neg, 1=Neu, 2=Pos
    groups   : (N,)     str     — subject ID
    sessions : (N,)     str     — session ID
    """
    mat_files = sorted(
        p for p in feature_dir.glob("*.mat") if p.name != "label.mat"
    )
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {feature_dir}")

    raw_labels = loadmat(feature_dir / "label.mat").get("label")
    if raw_labels is None or raw_labels.size != 15:
        raise ValueError("Could not read 15 trial labels from label.mat")
    trial_labels = np.array(
        [LABEL_MAP[int(v)] for v in raw_labels.ravel()], dtype=np.int64
    )

    features, labels, groups, sessions = [], [], [], []
    for mat_path in mat_files:
        subject_id = mat_path.stem.split("_", maxsplit=1)[0]
        session_id = mat_path.stem
        data = loadmat(mat_path)
        for trial_idx in range(15):
            parts = []
            for key_prefix in FEATURE_KEYS:
                key = f"{key_prefix}{trial_idx + 1}"
                if key not in data:
                    raise ValueError(f"{mat_path.name}: missing key '{key}'")
                arr = data[key]  # (channels, T, 5)
                T = arr.shape[1]
                parts.append(arr.transpose(1, 0, 2).reshape(T, -1))
            combined = np.concatenate(parts, axis=1).astype(np.float32)
            features.append(combined)
            labels.append(
                np.full(T, trial_labels[trial_idx], dtype=np.int64)
            )
            groups.append(np.full(T, subject_id))
            sessions.append(np.full(T, session_id))

    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(groups),
        np.concatenate(sessions),
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────


def normalize_by_session(
    features: np.ndarray, sessions: np.ndarray
) -> np.ndarray:
    """Label-free per-session z-score — removes cross-session DC offsets."""
    out = np.empty_like(features)
    for sid in np.unique(sessions):
        mask = sessions == sid
        block = features[mask]
        out[mask] = (block - block.mean(0)) / (block.std(0) + 1e-6)
    return out


def temporal_smooth(probs_or_preds: np.ndarray, window: int = 15) -> np.ndarray:
    """Sliding-window temporal filter — smooths noisy window-level predictions.

    If 2D probability distribution (N, C): filters each class posterior continuously.
    If 1D class labels (N,): converts to one-hot before uniform filtering.
    Takes argmax across classes after temporal filtering.
    """
    if window <= 1:
        return probs_or_preds.argmax(axis=1) if probs_or_preds.ndim == 2 else probs_or_preds

    if probs_or_preds.ndim == 2:
        smoothed = np.empty_like(probs_or_preds, dtype=np.float64)
        for c in range(probs_or_preds.shape[1]):
            smoothed[:, c] = uniform_filter1d(
                probs_or_preds[:, c].astype(np.float64), size=window, mode="nearest"
            )
        return smoothed.argmax(axis=1)
    else:
        one_hot = np.eye(len(CLASS_NAMES))[probs_or_preds]
        for c in range(len(CLASS_NAMES)):
            one_hot[:, c] = uniform_filter1d(
                one_hot[:, c].astype(np.float64), size=window, mode="nearest"
            )
        return one_hot.argmax(axis=1)


# ── Model ─────────────────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """FC → BN → ReLU → Dropout → FC → BN  +  skip projection."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)  # skip connection


class EmotionMLP(nn.Module):
    """MLP with residual blocks for stable deep training."""

    def __init__(self, input_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(512, dropout)
        self.down = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
        )
        self.res2 = ResidualBlock(256, dropout * 0.7)
        self.head = nn.Linear(256, len(CLASS_NAMES))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        return self.head(x)


# ── Mixup augmentation ────────────────────────────────────────────────────────


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate pairs of samples and their one-hot labels."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a = F.one_hot(y, len(CLASS_NAMES)).float()
    y_mix = lam * y_a + (1 - lam) * y_a[idx]
    return x_mix, y_mix


# ── Training loop ─────────────────────────────────────────────────────────────


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    patience: int,
    mixup_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Train MLP with mixup; returns (test_preds, test_probs)."""
    model = EmotionMLP(x_train.shape[1], dropout)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Warmup for 5 epochs then cosine decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - 5, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[5]
    )

    best_f1, stale, best_state = -1.0, 0, None
    for epoch in range(epochs):
        # ── train with mixup ──────────────────────────────────────────────
        model.train()
        for bx, by in loader:
            bx_mix, by_soft = mixup_batch(bx, by, alpha=mixup_alpha)
            optimizer.zero_grad()
            logits = model(bx_mix)
            # soft cross-entropy (works with non-integer labels from mixup)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(by_soft * log_probs).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(x_val)).argmax(1).numpy()
        val_f1 = f1_score(y_val, preds, average="macro")
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  "
                f"loss={loss.item():.4f}  val_F1={val_f1:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if stale >= patience:
            print(
                f"  Early stop @ epoch {epoch + 1}  "
                f"(best val F1={best_f1:.4f})"
            )
            break

    # ── test inference ────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        probs = F.softmax(logits, dim=1).numpy()
    return probs.argmax(1), probs


# ── Evaluation ────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{name} — held-out subjects")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.lower()}_confusion_matrix.png", dpi=180)
    plt.close(fig)


# ── Single-split evaluation ──────────────────────────────────────────────────


def run_single_split(args) -> None:
    """Train and evaluate on a single held-out subject split."""
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load features ─────────────────────────────────────────────────
    print("Loading SEED features (de_LDS + dasm_LDS + rasm_LDS) …")
    X, y, groups, sessions = load_seed_features(args.features_dir)
    print(
        f"  {X.shape[0]:,} windows  ×  {X.shape[1]} features  "
        f"from {len(np.unique(groups))} subjects"
    )

    # ── 2. Per-session normalisation ──────────────────────────────────────
    X = normalize_by_session(X, sessions)

    # ── 3. Subject split ──────────────────────────────────────────────────
    test_mask = np.isin(groups, args.test_subjects)
    val_mask = groups == args.val_subject
    train_mask = ~test_mask & ~val_mask

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    print(
        f"  Train {len(X_train):,}  |  Val {len(X_val):,}  "
        f"|  Test {len(X_test):,}"
    )

    # ── 4. StandardScaler (train only) ────────────────────────────────────
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train).astype(np.float32)
    Xs_val = scaler.transform(X_val).astype(np.float32)
    Xs_test = scaler.transform(X_test).astype(np.float32)

    # ── 5a. SVM ───────────────────────────────────────────────────────────
    print("\n=== SVM (Linear + Platt calibration) ===")
    base_svm = LinearSVC(C=0.1, max_iter=5000, random_state=args.seed)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base_svm, cv=3)),
    ])
    svm.fit(X[train_mask | val_mask], y[train_mask | val_mask])
    svm_probs = svm.predict_proba(X_test)   # calibrated probabilities
    svm_pred = svm_probs.argmax(1)

    # ── 5b. MLP ───────────────────────────────────────────────────────────
    print("\n=== MLP (residual blocks + mixup) ===")
    mlp_pred, mlp_probs = train_mlp(
        Xs_train, y_train, Xs_val, y_val, Xs_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        mixup_alpha=args.mixup_alpha,
    )

    # ── 5c. Soft ensemble (MLP 60% + SVM 40%) ────────────────────────────
    ensemble_probs = 0.6 * mlp_probs + 0.4 * svm_probs
    ensemble_pred = ensemble_probs.argmax(1)

    # ── 5d. Temporal smoothing (sliding window over posterior probs) ──────
    print(f"\n=== Temporal smoothing (window={args.smooth_window}) ===")
    svm_pred_raw = svm_probs.argmax(1)
    mlp_pred_raw = mlp_probs.argmax(1)
    ensemble_pred_raw = ensemble_probs.argmax(1)

    svm_pred = temporal_smooth(svm_probs, window=args.smooth_window)
    mlp_pred = temporal_smooth(mlp_probs, window=args.smooth_window)
    ensemble_pred = temporal_smooth(ensemble_probs, window=args.smooth_window)

    for tag, raw, smoothed in [
        ("SVM", svm_pred_raw, svm_pred),
        ("MLP", mlp_pred_raw, mlp_pred),
        ("Ensemble", ensemble_pred_raw, ensemble_pred),
    ]:
        raw_acc = accuracy_score(y_test, raw)
        sm_acc = accuracy_score(y_test, smoothed)
        print(f"  {tag}: {raw_acc:.4f} → {sm_acc:.4f}  (+{sm_acc - raw_acc:.4f})")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    results = {
        "split": {
            "held_out_subjects": args.test_subjects,
            "validation_subject": args.val_subject,
            "train_windows": int(len(X_train)),
            "val_windows": int(len(X_val)),
            "test_windows": int(len(X_test)),
        },
        "features": (
            "de_LDS (310) + dasm_LDS (135) + rasm_LDS (135) = 580 dims"
        ),
        "normalization": "per-session z-score + train-fitted StandardScaler",
        "temporal_smoothing": f"continuous posterior averaging, window={args.smooth_window}",
        "svm_raw": compute_metrics(y_test, svm_pred_raw),
        "mlp_raw": compute_metrics(y_test, mlp_pred_raw),
        "ensemble_raw": compute_metrics(y_test, ensemble_pred_raw),
        "svm": compute_metrics(y_test, svm_pred),
        "mlp": compute_metrics(y_test, mlp_pred),
        "ensemble": compute_metrics(y_test, ensemble_pred),
    }

    for name, pred in [
        ("SVM", svm_pred), ("MLP", mlp_pred), ("Ensemble", ensemble_pred)
    ]:
        save_confusion_matrix(y_test, pred, name, args.output_dir)
        m = results[name.lower()]
        print(
            f"\n{name}:  accuracy = {m['accuracy']:.4f}   "
            f"macro F1 = {m['macro_f1']:.4f}"
        )

    # ── 7. Per-subject accuracy breakdown ─────────────────────────────────
    print("\n=== Per-subject accuracy (Ensemble + smoothing) ===")
    test_groups = groups[test_mask]
    per_subject = {}
    for subj in sorted(np.unique(test_groups)):
        subj_mask = test_groups == subj
        subj_acc = accuracy_score(y_test[subj_mask], ensemble_pred[subj_mask])
        subj_f1 = f1_score(
            y_test[subj_mask], ensemble_pred[subj_mask], average="macro"
        )
        per_subject[subj] = {"accuracy": round(subj_acc, 4), "macro_f1": round(subj_f1, 4)}
        print(f"  Subject {subj}:  acc = {subj_acc:.4f}   F1 = {subj_f1:.4f}")
    results["per_subject"] = per_subject

    # ── 8. Classification report ──────────────────────────────────────────
    print("\n=== Classification Report (Ensemble + smoothing) ===")
    report = classification_report(
        y_test, ensemble_pred, target_names=CLASS_NAMES
    )
    print(report)
    results["classification_report"] = classification_report(
        y_test, ensemble_pred, target_names=CLASS_NAMES, output_dict=True
    )

    # ── 9. t-SNE visualisation ────────────────────────────────────────────
    print("Generating t-SNE visualisation …")
    n_sample = min(5000, len(Xs_test))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(Xs_test), n_sample, replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, max_iter=1000)
    emb = tsne.fit_transform(Xs_test[idx])

    fig, ax = plt.subplots(figsize=(7, 6))
    colours = ["#d62728", "#7f7f7f", "#2ca02c"]   # red=Neg, grey=Neu, green=Pos
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask_c = y_test[idx] == cls_idx
        ax.scatter(
            emb[mask_c, 0], emb[mask_c, 1],
            c=colours[cls_idx], label=cls_name,
            s=6, alpha=0.45, edgecolors="none",
        )
    ax.legend(fontsize=10, markerscale=4)
    ax.set_title("t-SNE of Test Features (colour = true label)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tsne_features.png", dpi=180)
    plt.close(fig)

    # ── 10. Save ──────────────────────────────────────────────────────────
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    print(f"\nSaved → {args.output_dir.resolve()}")


# ── LOSO cross-validation ────────────────────────────────────────────────────


def run_loso(args) -> None:
    """Leave-One-Subject-Out CV — the gold-standard SEED evaluation."""
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SEED features for LOSO CV …")
    X, y, groups, sessions = load_seed_features(args.features_dir)
    X = normalize_by_session(X, sessions)
    subjects = sorted(np.unique(groups))
    print(f"  {len(subjects)} subjects, {X.shape[0]:,} total windows\n")

    all_metrics = []
    for i, test_subj in enumerate(subjects):
        # Val = next subject cyclically
        val_subj = subjects[(i + 1) % len(subjects)]

        test_mask = groups == test_subj
        val_mask = groups == val_subj
        train_mask = ~test_mask & ~val_mask

        X_tr, X_v, X_te = X[train_mask], X[val_mask], X[test_mask]
        y_tr, y_v, y_te = y[train_mask], y[val_mask], y[test_mask]

        scaler = StandardScaler()
        Xs_tr = scaler.fit_transform(X_tr).astype(np.float32)
        Xs_v = scaler.transform(X_v).astype(np.float32)
        Xs_te = scaler.transform(X_te).astype(np.float32)

        # SVM
        base_svm = LinearSVC(C=0.1, max_iter=5000, random_state=args.seed)
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base_svm, cv=3)),
        ])
        svm_pipe.fit(X[train_mask | val_mask], y[train_mask | val_mask])
        svm_probs = svm_pipe.predict_proba(X_te)

        # MLP
        _, mlp_probs = train_mlp(
            Xs_tr, y_tr, Xs_v, y_v, Xs_te,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, dropout=args.dropout,
            patience=args.patience, mixup_alpha=args.mixup_alpha,
        )

        # Ensemble + smoothing
        ens_probs = 0.6 * mlp_probs + 0.4 * svm_probs
        ens_pred = temporal_smooth(ens_probs, window=args.smooth_window)

        fold_metrics = compute_metrics(y_te, ens_pred)
        all_metrics.append(fold_metrics)
        print(
            f"  [{i+1:>2}/{len(subjects)}] Subject {test_subj}:  "
            f"acc={fold_metrics['accuracy']:.4f}  "
            f"F1={fold_metrics['macro_f1']:.4f}"
        )

    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["macro_f1"] for m in all_metrics]
    loso_results = {
        "protocol": "LOSO (Leave-One-Subject-Out)",
        "n_folds": len(subjects),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "mean_macro_f1": round(float(np.mean(f1s)), 4),
        "std_macro_f1": round(float(np.std(f1s)), 4),
        "per_fold": {
            subj: m for subj, m in zip(subjects, all_metrics)
        },
    }
    print(
        f"\n{'='*50}\n"
        f"LOSO RESULT:  {np.mean(accs):.2%} ± {np.std(accs):.2%}  "
        f"(macro F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f})\n"
        f"{'='*50}"
    )
    (args.output_dir / "loso_results.json").write_text(
        json.dumps(loso_results, indent=2) + "\n"
    )
    print(f"Saved → {args.output_dir / 'loso_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEED EEG Emotion Classifier — SVM + MLP ensemble"
    )
    parser.add_argument(
        "--features-dir", type=Path, required=True,
        help="Path to SEED ExtractedFeatures directory"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--test-subjects", nargs="+", default=["13", "14", "15"]
    )
    parser.add_argument("--val-subject", default="12")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smooth-window", type=int, default=15,
        help="Sliding window size in seconds for posterior temporal smoothing"
    )
    parser.add_argument(
        "--loso", action="store_true",
        help="Run Leave-One-Subject-Out CV instead of single split"
    )
    args = parser.parse_args()

    if args.loso:
        run_loso(args)
    else:
        run_single_split(args)


if __name__ == "__main__":
    main()

