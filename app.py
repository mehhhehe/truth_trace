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

from flask import Flask, render_template, jsonify
import torch
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.datasets import UPFD

from model import BaselineClassifier, GATClassifier


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

    # Load UPFD splits
    train_dataset = UPFD(root=root, name=dataset_name, feature=feature_type, split="train")
    val_dataset = UPFD(root=root, name=dataset_name, feature=feature_type, split="val")
    test_dataset = UPFD(root=root, name=dataset_name, feature=feature_type, split="test")
    # Combine all splits for inference and display
    full_dataset: List = []
    full_dataset.extend(list(train_dataset))
    full_dataset.extend(list(val_dataset))
    full_dataset.extend(list(test_dataset))
    # Determine embedding dimension (length of node feature vectors)
    embed_dim = full_dataset[0].x.size(1) if len(full_dataset) > 0 else 0

    # Load baseline model
    baseline_model = None
    baseline_path = os.path.join(model_dir, "baseline_classifier.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            baseline_model = pickle.load(f)

    # Load GAT model
    gat_model = None
    gat_path = os.path.join(model_dir, "gat_classifier.pth")
    if os.path.exists(gat_path) and embed_dim > 0:
        gat_model = GATClassifier(input_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        gat_model.load_state_dict(torch.load(gat_path, map_location=device))
        gat_model.to(device)
        gat_model.eval()

    # Map a generated news ID to its corresponding Data object
    # Use enumeration index as ID (string) for simplicity
    id_to_data: Dict[str, object] = {str(i): data for i, data in enumerate(full_dataset)}

    @app.route("/")
    def index() -> str:
        news_ids = list(id_to_data.keys())
        return render_template("index.html", news_ids=news_ids)

    @app.route("/data")
    def data_endpoint() -> str:
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
        baseline_prob = None
        if baseline_model is not None and embed_dim > 0:
            root_vec = data.x[0].numpy()[:embed_dim]
            baseline_prob = float(baseline_model.predict_proba(root_vec.reshape(1, -1))[0])
        # GAT prediction
        gat_prob = None
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
        if news_id not in id_to_data:
            return jsonify({"error": "News ID not found"}), 404
        data = id_to_data[news_id]
        nodes: List[Dict] = []
        links: List[Dict] = []
        # root is node 0
        for idx in range(data.x.size(0)):
            nodes.append({
                "id": str(idx),
                "group": "news" if idx == 0 else "user",
            })
        # Build edges from edge_index
        for src, dst in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
            links.append({"source": str(src), "target": str(dst)})
        return jsonify({"nodes": nodes, "links": links})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)