"""
Training script for TruthTrace models using the PyTorch Geometric UPFD dataset.

This script trains both a text‑only baseline classifier and a Graph
Attention Network (GAT) on news propagation cascades.  Instead of
reading a bespoke JSON file, it loads the official
`UPFD` dataset via PyTorch Geometric, which automatically
downloads and caches the selected split and feature type.  Each
instance in the dataset represents a single cascade with a root
`news` node and associated user nodes.  Node features are provided by
the chosen feature type (e.g., BERT or content embeddings).  The
dataset comes with predefined train, validation and test splits so
manual splitting is unnecessary.
"""

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


def train_baseline(train_dataset: List, val_dataset: List, embed_dim: int, model_dir: str) -> BaselineClassifier:
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
    Train a GAT model on the cascade‑level classification task using the UPFD dataset.

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
        Mini‑batch size.
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
    # Create DataLoaders
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
    # Evaluate on test set
    test_acc = evaluate(model, test_loader, device)
    print(f"GAT test accuracy: {test_acc:.4f}")
    # Save model
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
        "--dataset", type=str, choices=["politifact", "gossipcop"], default="politifact",
        help="Which UPFD sub‑dataset to use (politifact or gossipcop)"
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
    # Train GAT model
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