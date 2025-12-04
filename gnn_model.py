"""
Model definitions for the TruthTrace full‑batch implementation.

This module defines two classifiers: a simple text‑only baseline and a
graph neural network based on Graph Attention Networks (GAT).  The
baseline operates on TF–IDF vectors extracted from each news item,
ignoring the graph structure entirely.  The GAT model processes
propagation graphs to capture both the content and the diffusion
patterns, aggregating information from neighbouring nodes through
attention mechanisms【453453708915908†L106-L123】.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool  # type: ignore


class BaselineClassifier(nn.Module):
    """A minimal logistic regression model for fake news detection.

    The classifier consists of a single fully connected layer mapping
    the input feature vector (typically TF–IDF) to two output logits.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of input vectors."""
        return self.fc(x)


class GATClassifier(nn.Module):
    """Two‑layer Graph Attention Network for graph classification.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_channels : int
        Hidden dimension for the intermediate representations.
    heads : int, optional
        Number of attention heads used in each GAT layer.  Using
        multiple heads increases model expressiveness at the cost of
        memory.  Default is 4.
    dropout : float, optional
        Dropout probability applied after each layer (default 0.3).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.dropout = dropout
        # First GAT layer expands dimensions by number of heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer reduces back to hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        # Classification head
        self.lin = nn.Linear(hidden_channels * heads, 2)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GAT model.

        The input is a ``torch_geometric.data.Data`` object with
        attributes ``x`` (node features), ``edge_index`` (graph
        connectivity) and ``batch`` (node‑to‑graph assignment).  The
        output is a tensor of shape ``[batch_size, 2]`` with raw
        logits for each graph in the batch.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Global mean pooling
        x = global_mean_pool(x, batch)
        # Classification
        return self.lin(x)