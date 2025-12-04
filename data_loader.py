"""
Data loading and preprocessing for TruthTrace.

This module defines classes and functions to load a JSON dataset of news
propagation cascades, compute text embeddings with a transformer model,
construct graphs with user and news nodes, and prepare PyTorch
Geometric `Data` objects.  The implementation follows the high‑level
workflow: build graphs from inputs, extract features for each node and
return objects suitable for GNN training【126310739344574†L125-L141】.

Each JSON record must include a unique news ID, the news text, a
binary label (0 = real, 1 = fake) and a list of user objects.  Each
user object has an ID, the ID of the parent node it retweeted from,
and optional metadata (e.g. follower count and account age).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore


class TextEmbedder:
    """Wrapper around HuggingFace transformer models to produce text embeddings.

    If the transformers library is unavailable, this class raises
    informative errors.  Otherwise it loads a pre‑trained model and
    tokenizer and exposes an `encode` method to convert raw text into
    a fixed‑size embedding vector (using the [CLS] token as the
    representation).
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "The transformers library is required to compute text embeddings. "
                "Install it via `pip install transformers` before proceeding."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Put the model in evaluation mode to disable dropout
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """Return a numpy vector representing the input text.

        This uses the pooled output of the model (the [CLS] token).  The
        returned vector has dimension equal to the hidden size of the
        underlying transformer (typically 768 for BERT‑base).
        """
        # Tokenize and truncate to the model's max length
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
        # We take the first token ([CLS]) from the first (and only) batch
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return cls_embedding.cpu().numpy()


@dataclass
class CascadeRecord:
    """Internal representation of a news propagation cascade."""
    news_id: str
    text: str
    label: int
    users: List[Dict]


class TruthTraceDataset:
    """
    Load a dataset of news cascades and construct PyTorch Geometric Data objects.

    Each cascade is converted into a heterogenous graph with one news
    node and multiple user nodes.  Node features include a BERT
    embedding for the news text and simple numerical features for each
    user (normalised follower count, account age, in‑degree and
    out‑degree).  Edges connect users to the node they retweeted.

    The resulting `Data` objects have attributes:

    - `x` (torch.Tensor): Feature matrix of shape (num_nodes, feature_dim).
    - `edge_index` (torch.LongTensor): 2×num_edges tensor defining edge
      connectivity (in COO format).
    - `y` (torch.Tensor): Graph label (0 or 1) indicating real or fake.
    - `news_idx` (int): Index of the news node within the node list.
    - `id` (str): News ID.

    Splits for training/validation/test are generated via
    `train_val_test_split`.
    """

    def __init__(
        self,
        data_file: str,
        embedder: Optional[TextEmbedder] = None,
        user_feature_stats: bool = True,
    ) -> None:
        self.data_file = data_file
        self.embedder = embedder
        self.records: List[CascadeRecord] = []
        self.graphs: List[Data] = []
        # Load data
        self._load_records()
        # Compute statistics to normalise user features
        self._prepare_statistics()
        # Build graphs
        self._build_graphs()

    def _load_records(self) -> None:
        with open(self.data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for entry in raw:
            rec = CascadeRecord(
                news_id=entry["id"],
                text=entry["text"],
                label=int(entry["label"]),
                users=entry.get("users", []),
            )
            self.records.append(rec)

    def _prepare_statistics(self) -> None:
        """Compute min and max values of user metadata for normalisation."""
        follower_counts = []
        account_ages = []
        for rec in self.records:
            for user in rec.users:
                feat = user.get("features", {})
                follower_counts.append(feat.get("follower_count", 0))
                account_ages.append(feat.get("account_age_days", 0))
        # Avoid division by zero: if no users, set to 1
        self.follow_min = float(min(follower_counts)) if follower_counts else 0.0
        self.follow_max = float(max(follower_counts)) if follower_counts else 1.0
        self.age_min = float(min(account_ages)) if account_ages else 0.0
        self.age_max = float(max(account_ages)) if account_ages else 1.0

    def _normalise(self, value: float, v_min: float, v_max: float) -> float:
        if v_max == v_min:
            return 0.0
        return (value - v_min) / (v_max - v_min)

    def _build_graphs(self) -> None:
        for rec in self.records:
            G = nx.DiGraph()
            # Add news node first
            news_idx = 0
            # Each node will be assigned an incremental index
            node_features: List[np.ndarray] = []
            # Compute text embedding for news
            if self.embedder is not None:
                text_emb = self.embedder.encode(rec.text)
            else:
                # If no embedder provided, use random vector (for demonstration)
                text_emb = np.random.randn(768)
            # Placeholder for user feature dimension; we will append user features later
            user_feat_len = 4  # follower_count, account_age, in_degree, out_degree
            news_feature = np.concatenate([
                text_emb,
                np.zeros(user_feat_len, dtype=float)
            ])
            G.add_node(rec.news_id, type="news")
            node_features.append(news_feature)

            # Add user nodes and record edges
            for user in rec.users:
                uid = user["id"]
                parent = user.get("parent", rec.news_id)
                G.add_node(uid, type="user")
                # We'll add edge from parent to this user
                G.add_edge(parent, uid)
                # Build initial feature vector: zeros for text embedding, will append user features later
                user_vector = np.concatenate([
                    np.zeros_like(text_emb),
                    np.zeros(user_feat_len, dtype=float)
                ])
                node_features.append(user_vector)
            # At this stage node order in node_features matches order of G.nodes
            # Compute in/out degrees for structural features
            # Create mapping from node id to index in node_features
            id_to_index = {nid: idx for idx, nid in enumerate(G.nodes())}
            # Precompute degrees
            in_degrees = G.in_degree()
            out_degrees = G.out_degree()
            # Fill user features
            updated_features = []
            for idx, (nid, data) in enumerate(G.nodes(data=True)):
                vec = node_features[idx]
                # Determine if news node or user
                if data.get("type") == "news":
                    # Already filled with text embedding and zeros for user features
                    updated_features.append(vec)
                else:
                    # Fill user features: follower count, account age
                    # Use normalised values
                    user_obj = next(u for u in rec.users if u["id"] == nid)
                    feat = user_obj.get("features", {})
                    follower = feat.get("follower_count", 0)
                    age = feat.get("account_age_days", 0)
                    follower_norm = self._normalise(follower, self.follow_min, self.follow_max)
                    age_norm = self._normalise(age, self.age_min, self.age_max)
                    # Degree features
                    indeg = in_degrees[nid]
                    outdeg = out_degrees[nid]
                    # Compose final user features
                    user_feats = np.array([follower_norm, age_norm, float(indeg), float(outdeg)], dtype=float)
                    # Put them at the end of the vector
                    new_vec = np.concatenate([
                        vec[: len(text_emb)],
                        user_feats
                    ])
                    updated_features.append(new_vec)
            # Build edge_index tensor (convert directed graph to undirected for GNN)
            # We'll make edges bidirectional to allow message passing both ways
            edges = list(G.edges())
            # Add reverse edges
            edges += [(dst, src) for src, dst in edges]
            # Convert node id pairs to indices
            edge_index = torch.tensor([
                [id_to_index[src] for src, dst in edges],
                [id_to_index[dst] for src, dst in edges],
            ], dtype=torch.long)
            x = torch.tensor(np.stack(updated_features), dtype=torch.float)
            y = torch.tensor([rec.label], dtype=torch.long)
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                news_idx=0,
                id=rec.news_id,
            )
            self.graphs.append(data)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

    def train_val_test_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return indices for train/validation/test splits.

        The split is stratified by label to preserve class ratios.  The
        returned values are lists of indices into `self.graphs`.
        """
        labels = [int(data.y.item()) for data in self.graphs]
        all_indices = list(range(len(self.graphs)))
        # First split into train+val and test
        train_val_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
        # Compute new labels for train_val
        train_val_labels = [labels[i] for i in train_val_indices]
        # Split train_val into train and val
        val_relative_size = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_relative_size,
            stratify=train_val_labels,
            random_state=random_state,
        )
        return train_indices, val_indices, test_indices


def collate_batch(batch: List[Data]) -> Data:
    """Collate function for torch_geometric DataLoader.

    This simply merges a list of graphs into a mini‑batch.  Since each
    graph is independent there is no special processing needed.
    """
    return DataLoader.collate(batch)


def get_data_loader(
    dataset: TruthTraceDataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Return a PyTorch Geometric DataLoader for the given subset indices."""
    subset = [dataset[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
