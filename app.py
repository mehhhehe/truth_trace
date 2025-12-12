
"""
Flask web application for TruthTrace visualisation and inference using the UPFD dataset.

This server loads the UPFD dataset and trained models, exposes endpoints
for fetching predictions and graph structures, and renders an
interactive dashboard.  The dashboard allows a user to select a news
cascade, view its propagation graph, and compare probabilities from
the baseline and GAT models.
"""

import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from flask import Flask, jsonify, render_template
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.datasets import UPFD as BaseUPFD

from model import BaselineClassifier, GATClassifier


class LocalUPFD(BaseUPFD):
    """
    Local-only UPFD dataset.

    This subclass disables the built-in Google Drive download step since the links in the library are broken and
    assumes that the raw UPFD files have already been downloaded and
    extracted into:

        root / <name> / raw /

    with the following files present:

        - node_graph_id.npy
        - graph_labels.npy
        - A.txt
        - train_idx.npy
        - val_idx.npy
        - test_idx.npy
        - new_<feature>_feature.npz  (e.g. new_bert_feature.npz)

    If these files are missing, this will raise a RuntimeError instead
    of silently calling Google Drive.
    """

    def download(self) -> None:  # type: ignore[override]
        raw_dir = self.raw_dir
        missing = [
            fname
            for fname in self.raw_file_names
            if not os.path.exists(os.path.join(raw_dir, fname))
        ]
        if missing:
            raise RuntimeError(
                f"LocalUPFD expected existing raw files in '{raw_dir}', "
                f"but the following are missing: {missing}. "
                "Place the manually downloaded UPFD raw files there."
            )
        # No-op otherwise: do NOT attempt any remote download.
        return


def create_app() -> Flask:
    app = Flask(__name__)

    # Configuration: root directory, dataset name, feature type and model directory
    root = os.environ.get("TRUTHTRACE_ROOT", "./data")
    dataset_name = os.environ.get("TRUTHTRACE_DATASET", "gossipcop")
    feature_type = os.environ.get("TRUTHTRACE_FEATURE", "bert")
    model_dir = os.environ.get("TRUTHTRACE_MODEL_DIR", "models")
    hidden_dim = int(os.environ.get("TRUTHTRACE_HIDDEN_DIM", 64))
    num_heads = int(os.environ.get("TRUTHTRACE_NUM_HEADS", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load UPFD splits from local raw data (no Google Drive)
    # This will:
    #   - check raw/<files> exist
    #   - process them into processed/<feature>/train.pt, val.pt, test.pt if needed
    train_dataset = LocalUPFD(
        root=root,
        name=dataset_name,
        feature=feature_type,
        split="train",
    )
    val_dataset = LocalUPFD(
        root=root,
        name=dataset_name,
        feature=feature_type,
        split="val",
    )
    test_dataset = LocalUPFD(
        root=root,
        name=dataset_name,
        feature=feature_type,
        split="test",
    )

    # Combine all splits for inference and display
    full_dataset: List = []
    full_dataset.extend(list(train_dataset))
    full_dataset.extend(list(val_dataset))
    full_dataset.extend(list(test_dataset))

    if not full_dataset:
        raise RuntimeError("Combined UPFD dataset is empty – check your data paths.")

    # Determine embedding dimension (length of node feature vectors)
    embed_dim = full_dataset[0].x.size(1)

    # Load baseline model (e.g. sklearn LogisticRegression)
    baseline_model = None
    baseline_path = os.path.join(model_dir, "baseline_classifier.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            baseline_model = pickle.load(f)

    # Load GAT model
    gat_model = None
    gat_path = os.path.join(model_dir, "gat_classifier.pth")
    if os.path.exists(gat_path) and embed_dim > 0:
        gat_model = GATClassifier(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        gat_model.load_state_dict(torch.load(gat_path, map_location=device))
        gat_model.to(device)
        gat_model.eval()

    # Map a generated news ID to its corresponding Data object
    id_to_data: Dict[str, object] = {str(i): data for i, data in enumerate(full_dataset)}

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    @app.route("/")
    def index() -> str:
        news_ids = list(id_to_data.keys())
        return render_template("index.html", news_ids=news_ids)

    @app.route("/data")
    def data_endpoint():
        data_list = []
        for news_id, data in id_to_data.items():
            label = int(data.y.item())
            data_list.append({"id": news_id, "label": label})
        return jsonify(data_list)

    @app.route("/predict/<news_id>")
    def predict(news_id: str):
        if news_id not in id_to_data:
            return jsonify({"error": "News ID not found"}), 404

        data = id_to_data[news_id]

        # Baseline prediction
        baseline_prob: float | None = None
        if baseline_model is not None and embed_dim > 0:
            # Root node is index 0 by construction in UPFD
            root_vec = data.x[0].cpu().numpy()[:embed_dim]
            probs = baseline_model.predict_proba(root_vec.reshape(1, -1))
            probs = np.asarray(probs)

            if probs.ndim == 2:
                # Standard sklearn-style: shape (n_samples, n_classes)
                if hasattr(baseline_model, "classes_") and 1 in baseline_model.classes_:
                    cls_idx = list(baseline_model.classes_).index(1)
                else:
                    cls_idx = probs.shape[1] - 1
                baseline_prob = float(probs[0, cls_idx])
            elif probs.ndim == 1:
                baseline_prob = float(probs[0])
            else:
                baseline_prob = float(probs.ravel()[0])

        # GAT prediction
        gat_prob: float | None = None
        if gat_model is not None:
            loader = GeoDataLoader([data], batch_size=1)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = gat_model(batch)
                    prob = torch.sigmoid(logits)[0].item()
                    gat_prob = float(prob)

        return jsonify({"baseline": baseline_prob, "gat": gat_prob})

    @app.route("/graph_json/<news_id>")
    def graph_json(news_id: str):
        """
        JSON API used by both the main dashboard and the standalone graph view.

        Returns:
          - nodes: list of {id, group}
          - links: list of {source, target}
          - label / label_name
          - optional text (if attached as data.text)
          - node_features: first K dims per node, for colour-by-similarity, etc.
        """
        if news_id not in id_to_data:
            return jsonify({"error": "News ID not found"}), 404

        data = id_to_data[news_id]
        nodes: List[Dict] = []
        links: List[Dict] = []

        # Root is node 0
        for idx in range(data.x.size(0)):
            nodes.append(
                {
                    "id": str(idx),
                    "group": "news" if idx == 0 else "user",
                }
            )

        # Build edges from edge_index
        src_list = data.edge_index[0].tolist()
        dst_list = data.edge_index[1].tolist()
        for src, dst in zip(src_list, dst_list):
            links.append({"source": str(src), "target": str(dst)})

        # Graph label
        label_int = int(data.y.item())
        label_name = "fake" if label_int == 1 else "real"

        # Optional text attribute (will be None for raw UPFD)
        text = getattr(data, "text", None)

        # Node features: first K dims for each node (to keep JSON small)
        K = min(16, data.x.size(1))
        x_cpu = data.x.cpu().numpy()
        node_features: Dict[str, List[float]] = {}
        for idx in range(x_cpu.shape[0]):
            vec = x_cpu[idx][:K]
            node_features[str(idx)] = [float(v) for v in vec]

        return jsonify(
            {
                "nodes": nodes,
                "links": links,
                "label": label_int,
                "label_name": label_name,
                "text": text,
                "node_features": node_features,
            }
        )

    @app.route("/graph/<news_id>")
    def graph_view(news_id: str):
        if news_id not in id_to_data:
            return "News ID not found", 404
        return render_template("graph.html", news_id=news_id)

    


    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
