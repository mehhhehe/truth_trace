"""
Model definitions for TruthTrace.

This module defines two models:

1. **BaselineClassifier** — a simple text‑only classifier that
   operates on BERT embeddings of news articles.  It uses scikit‑learn's
   logistic regression to predict fake vs. real labels【126310739344574†L246-L252】.

2. **GATClassifier** — a graph neural network model based on the
   Graph Attention Network (GAT) architecture.  It processes
   heterogenous graphs of news and user nodes and outputs
   probabilities of misinformation for each graph【126310739344574†L125-L141】.

Both models expose a `.fit()` method for training and a `.predict_proba()`
method for obtaining probabilities on new examples.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaselineClassifier:
    """Text‑only baseline using logistic regression.

    This class wraps a scikit‑learn logistic regression model.  It
    expects input arrays of shape (n_samples, embedding_dim) and
    produces probabilities for the positive class (fake news)【126310739344574†L246-L252】.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=max_iter)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Standardise features to zero mean and unit variance
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("BaselineClassifier has not been fitted.")
        X_scaled = self.scaler.transform(X)
        # Return probability of positive class
        return self.clf.predict_proba(X_scaled)[:, 1]


class GATClassifier(nn.Module):
    """Graph Attention Network for cascade‑level classification.

    The model uses two GAT layers with multi‑head attention, followed by
    global mean pooling to aggregate node embeddings and a final linear
    layer for binary classification【126310739344574†L125-L141】.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        # The first GAT layer maps from input_dim to hidden_dim
        self.gat1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True, dropout=0.2)
        # The second GAT layer reduces concatenated heads to hidden_dim
        self.gat2 = GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=1, concat=True, dropout=0.2)
        # Classification head
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        """Forward pass returning raw logits for each graph.

        `data` should be a `torch_geometric.data.Data` object or a
        mini‑batch thereof.  The function expects `data.x`,
        `data.edge_index` and `data.batch` attributes.  It returns a
        tensor of shape (batch_size,) containing logits for the positive
        class.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        # Pool node embeddings to graph embedding
        x = global_mean_pool(x, batch)
        # Linear layer outputs logits
        logits = self.lin(x).squeeze(-1)
        return logits

    def predict_proba(self, loader: Iterable) -> np.ndarray:
        """Return predicted probabilities for each graph in the loader.

        The loader should yield mini‑batches of Data objects.  The model
        must be in evaluation mode prior to calling this method.
        """
        self.eval()
        probs: List[float] = []
        with torch.no_grad():
            for data in loader:
                data = data.to(next(self.parameters()).device)
                logits = self(data)
                prob = torch.sigmoid(logits)
                probs.extend(prob.cpu().numpy().tolist())
        return np.array(probs)
