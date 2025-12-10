"""
Evaluate saved TruthTrace models on the UPFD test set.

This script:
- Loads the UPFD test split (same root/dataset/feature as training)
- Loads the *existing* baseline and GAT models from --model_dir
- Computes accuracy, precision, recall, F1, ROC-AUC
- Saves metrics as JSON
- Saves clean confusion matrix images (PNG)

Expected in --model_dir:
- baseline_classifier.pkl   (pickle file)
- gat_classifier.pth        (state_dict for GAT model)
"""

import argparse
import os
import pickle
import json
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.datasets import UPFD

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from model import BaselineClassifier, GATClassifier


# ===================== Helpers ===================== #

def extract_root_embeddings(dataset: List, embed_dim: int) -> np.ndarray:
    """
    Return a matrix of shape (n_samples, embed_dim) with node embeddings for the news roots.

    In the UPFD dataset, the root node (the news post) is always the first node.
    """
    embs: List[np.ndarray] = []
    for data in dataset:
        vec = data.x[0].numpy()
        embs.append(vec[:embed_dim])
    return np.stack(embs)


def extract_labels(dataset: List) -> np.ndarray:
    """Return an array of labels (0/1) for each graph in the dataset."""
    return np.array([int(data.y.item()) for data in dataset])


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    out_path: str,
) -> None:
    """
    Render a confusion matrix as a PNG with clean layout.

    - Colorbar to the right with padding
    - Title with padding so it doesn’t collide with colorbar
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, interpolation="nearest")

    # Colorbar with spacing so it doesn't overlap
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, pad=14)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # Annotate each cell with counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_and_save_metrics_and_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray],
    json_out_path: str,
    cm_png_out_path: str,
    title: str,
) -> None:
    """
    Compute metrics and save:
    - JSON file with accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    - Confusion matrix image (PNG)
    """
    metrics = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ROC-AUC (if scores exist and both classes present)
    if y_scores is not None and len(y_scores) == len(y_true):
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)

    with open(json_out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        cm,
        classes=["Real (0)", "Fake (1)"],
        title=title,
        out_path=cm_png_out_path,
    )


# ===================== Baseline evaluation ===================== #

def evaluate_baseline(
    test_dataset: List,
    model_dir: str,
) -> None:
    """
    Load baseline model from disk and evaluate on test set.

    Saves in model_dir:
      - baseline_metrics_test.json
      - baseline_confusion_matrix.png
    """
    model_path = os.path.join(model_dir, "baseline_classifier.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    with open(model_path, "rb") as f:
        baseline: BaselineClassifier = pickle.load(f)

    embed_dim = test_dataset[0].x.size(1)
    X_test = extract_root_embeddings(test_dataset, embed_dim)
    y_test = extract_labels(test_dataset)

    probs = baseline.predict_proba(X_test)
    y_pred = (probs >= 0.5).astype(int)

    metrics_path = os.path.join(model_dir, "baseline_metrics_test.json")
    cm_png_path = os.path.join(model_dir, "baseline_confusion_matrix.png")

    compute_and_save_metrics_and_cm(
        y_true=y_test,
        y_pred=y_pred,
        y_scores=probs,
        json_out_path=metrics_path,
        cm_png_out_path=cm_png_path,
        title="Baseline Confusion Matrix (Test Set)",
    )

    print(f"[Baseline] Saved metrics → {metrics_path}")
    print(f"[Baseline] Saved confusion matrix → {cm_png_path}")


# ===================== GAT evaluation ===================== #

def evaluate_gat(
    test_dataset: List,
    model_dir: str,
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Load GAT model from disk and evaluate on test set.

    Saves in model_dir:
      - gat_metrics_test.json
      - gat_confusion_matrix.png

    NOTE: hidden_dim and num_heads must match what you used when training.
    """
    model_path = os.path.join(model_dir, "gat_classifier.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GAT model not found: {model_path}")

    input_dim = test_dataset[0].x.size(1)

    model = GATClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = batch.y.view(-1).cpu().numpy().astype(int)

            all_scores.append(probs)
            all_labels.append(labels)

    if not all_labels:
        print("[GAT] Test loader empty; no metrics computed.")
        return

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    metrics_path = os.path.join(model_dir, "gat_metrics_test.json")
    cm_png_path = os.path.join(model_dir, "gat_confusion_matrix.png")

    compute_and_save_metrics_and_cm(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        json_out_path=metrics_path,
        cm_png_out_path=cm_png_path,
        title="GAT Confusion Matrix (Test Set)",
    )

    print(f"[GAT] Saved metrics → {metrics_path}")
    print(f"[GAT] Saved confusion matrix → {cm_png_path}")


# ===================== Main ===================== #

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved TruthTrace models on UPFD test set")
    parser.add_argument(
        "--root", type=str, required=True,
        help="Directory where the UPFD dataset is stored/downloaded (same as training)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["politifact", "gossipcop"], default="politifact",
        help="Which UPFD sub-dataset to use (must match training)"
    )
    parser.add_argument(
        "--feature", type=str, choices=["bert", "content", "spacy", "profile"], default="bert",
        help="Node feature type (must match training)"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing saved models (baseline_classifier.pkl, gat_classifier.pth)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64,
        help="Hidden dimension used for GAT (must match training)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4,
        help="Number of attention heads for GAT (must match training)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for GAT evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load ONLY the test split
    test_dataset = UPFD(
        root=args.root,
        name=args.dataset,
        feature=args.feature,
        split="test",
    )
    print(f"Loaded UPFD test split: {len(test_dataset)} graphs")

    # Evaluate baseline (uses existing .pkl)
    evaluate_baseline(test_dataset, args.model_dir)

    # Evaluate GAT (uses existing .pth)
    evaluate_gat(
        test_dataset=test_dataset,
        model_dir=args.model_dir,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
