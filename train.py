
import argparse
import os
import pickle
from typing import List

import numpy as np
import torch
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.datasets import UPFD

from tqdm import tqdm

from model import BaselineClassifier, GATClassifier

# Metrics and plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def extract_root_embeddings(dataset: List, embed_dim: int) -> np.ndarray:
    """
    Return a matrix of shape (n_samples, embed_dim) with node embeddings for the news roots.

    In the UPFD dataset the root node (the news post) is always the first node
    in the graph.  Its feature vector is of length `embed_dim`.
    """
    embs: List[np.ndarray] = []
    for data in dataset:
        # root node is at index 0
        vec = data.x[0].numpy()
        # If the feature vector is larger than embed_dim (shouldn't happen for UPFD
        # since the feature length matches embed_dim), slice accordingly.
        embs.append(vec[:embed_dim])
    return np.stack(embs)


def extract_labels(dataset: List) -> np.ndarray:
    """Return an array of labels (0/1) for each graph in the dataset."""
    return np.array([int(data.y.item()) for data in dataset])


def compute_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None,
    json_out_path: str | None = None,
    cm_png_out_path: str | None = None,
    title: str = "Confusion matrix",
) -> None:
    """
    Compute standard binary classification metrics and optionally save them to JSON,
    and render the confusion matrix as a PNG image.

    Metrics:
    - accuracy
    - precision, recall, f1 (binary)
    - confusion_matrix (2x2 list [[tn, fp], [fn, tp]])
    - roc_auc (if computable, else None)
    """
    # Core metrics
    metrics: dict[str, float | list | None] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ROC-AUC
    if y_scores is not None and len(y_scores) == len(y_true):
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # Ensure output directory exists (if any file path is given)
    base_dir = None
    for p in (json_out_path, cm_png_out_path):
        if p:
            base_dir = os.path.dirname(p)
            break
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    # Save JSON metrics
    if json_out_path is not None:
        import json

        with open(json_out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Plot and save confusion matrix image
    if cm_png_out_path is not None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        classes = ["Real (0)", "Fake (1)"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        # Annotate cells with counts
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
        fig.savefig(cm_png_out_path, bbox_inches="tight")
        plt.close(fig)


def train_baseline(
    train_dataset: List,
    val_dataset: List,
    embed_dim: int,
    model_dir: str,
) -> BaselineClassifier:
    """
    Fit a baseline logistic regression model on the root embeddings of each cascade.

    Parameters
    ----------
    train_dataset, val_dataset : list of Data objects
        The training and validation sets loaded from UPFD.
    embed_dim : int
        The dimensionality of the node feature vectors.  For UPFD this
        is the length of the chosen feature type (e.g., 768 for BERT).
    model_dir : str
        Directory in which to save the fitted baseline model.
    """
    X_train = extract_root_embeddings(train_dataset, embed_dim)
    y_train = extract_labels(train_dataset)
    X_val = extract_root_embeddings(val_dataset, embed_dim)
    y_val = extract_labels(val_dataset)

    baseline = BaselineClassifier()
    baseline.fit(X_train, y_train)

    # Simple validation accuracy for monitoring
    val_probs = baseline.predict_proba(X_val)
    val_pred = (val_probs >= 0.5).astype(int)
    val_acc = (val_pred == y_val).mean()
    print(f"Baseline validation accuracy: {val_acc:.4f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "baseline_classifier.pkl"), "wb") as f:
        pickle.dump(baseline, f)

    return baseline


def evaluate(model: GATClassifier, loader: GeoDataLoader, device: torch.device) -> float:
    """
    Compute accuracy of the GAT model on data from a DataLoader.
    Primarily used for validation monitoring.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            labels = batch.y
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_gat(
    train_dataset: List,
    val_dataset: List,
    test_dataset: List,
    hidden_dim: int,
    num_heads: int,
    epochs: int,
    batch_size: int,
    model_dir: str,
    device: torch.device,
) -> GATClassifier:
    """
    Train a GAT model on the cascade-level classification task using the UPFD dataset.

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset : list of Data objects
        The splits of the UPFD dataset.  Each Data object has node
        features `x`, edge indices `edge_index`, a graph label `y`,
        and a `batch` attribute automatically added by the DataLoader.
    hidden_dim : int
        Hidden dimension for the GAT layers.
    num_heads : int
        Number of attention heads in the first GAT layer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    model_dir : str
        Directory to save the trained model.
    device : torch.device
        Device on which to run training.
    """
    # Determine input dimension from the first graph's feature matrix
    input_dim = train_dataset[0].x.size(1)
    model = GATClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCEWithLogitsLoss()

    # DataLoaders
    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        total_loss = 0.0
        total_samples = 0

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            labels = batch.y.float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Full test-set evaluation with metrics and confusion matrix image -----
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = batch.y.view(-1).cpu().numpy().astype(int)
            all_scores.append(probs)
            all_labels.append(labels)

    if all_labels:
        y_true = np.concatenate(all_labels)
        y_scores = np.concatenate(all_scores)
        y_pred = (y_scores >= 0.5).astype(int)
        test_acc = float((y_pred == y_true).mean())
    else:
        y_true = np.array([])
        y_scores = np.array([])
        y_pred = np.array([])
        test_acc = 0.0

    print(f"GAT test accuracy: {test_acc:.4f}")

    gat_metrics_path = os.path.join(model_dir, "gat_metrics_test.json")
    gat_cm_png_path = os.path.join(model_dir, "gat_confusion_matrix.png")
    compute_and_save_metrics(
        y_true,
        y_pred,
        y_scores,
        json_out_path=gat_metrics_path,
        cm_png_out_path=gat_cm_png_path,
        title="GAT Confusion Matrix (Test Set)",
    )
    print(f"Saved GAT test metrics to {gat_metrics_path}")
    print(f"Saved GAT confusion matrix image to {gat_cm_png_path}")

    # Save model weights
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "gat_classifier.pth"))

    return model


def main():
    parser = argparse.ArgumentParser(description="Train TruthTrace models using the UPFD dataset")
    parser.add_argument(
        "--root", type=str, required=True,
        help="Directory where the UPFD dataset should be stored.  If not present, it will be downloaded."
    )
    parser.add_argument(
        "--dataset", type=str, choices=["politifact", "gossipcop"], default="gossipcop",
        help="Which UPFD sub-dataset to use (politifact or gossipcop)"
    )
    parser.add_argument(
        "--feature", type=str, choices=["bert", "content", "spacy", "profile"], default="bert",
        help="Type of node features to load from the UPFD dataset"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs for the GAT model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64,
        help="Hidden dimension for the GAT model"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4,
        help="Number of attention heads in the first GAT layer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for GNN training"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cuda or cpu)"
    )
    args = parser.parse_args()

    # Load the UPFD dataset for each split
    train_dataset = UPFD(root=args.root, name=args.dataset, feature=args.feature, split="train")
    val_dataset = UPFD(root=args.root, name=args.dataset, feature=args.feature, split="val")
    test_dataset = UPFD(root=args.root, name=args.dataset, feature=args.feature, split="test")
    print(
        f"Loaded UPFD dataset '{args.dataset}' with feature '{args.feature}': "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    # Determine embedding dimension (all nodes share the same feature length)
    embed_dim = train_dataset[0].x.size(1)

    # Train baseline classifier
    print("Training baseline classifier...")
    baseline = train_baseline(train_dataset, val_dataset, embed_dim, args.model_dir)

    # Evaluate baseline on test set and produce metrics + confusion-matrix image
    print("Evaluating baseline classifier on test set...")
    X_test = extract_root_embeddings(test_dataset, embed_dim)
    y_test = extract_labels(test_dataset)
    test_probs = baseline.predict_proba(X_test)
    test_pred = (test_probs >= 0.5).astype(int)

    baseline_metrics_path = os.path.join(args.model_dir, "baseline_metrics_test.json")
    baseline_cm_png_path = os.path.join(args.model_dir, "baseline_confusion_matrix.png")
    compute_and_save_metrics(
        y_test,
        test_pred,
        test_probs,
        json_out_path=baseline_metrics_path,
        cm_png_out_path=baseline_cm_png_path,
        title="Baseline Confusion Matrix (Test Set)",
    )
    print(f"Saved baseline test metrics to {baseline_metrics_path}")
    print(f"Saved baseline confusion matrix image to {baseline_cm_png_path}")

    # Train GAT model (this will also compute and save its test metrics and confusion matrix)
    print("Training GAT classifier...")
    device = torch.device(args.device)
    _ = train_gat(
        train_dataset,
        val_dataset,
        test_dataset,
        args.hidden_dim,
        args.num_heads,
        args.epochs,
        args.batch_size,
        args.model_dir,
        device,
    )


if __name__ == "__main__":
    main()
