# TruthTrace Full Version (GAT on Full Propagation Trees)

This repository contains a complete implementation of **TruthTrace** for the
"full‑batch" setting.  It follows the high‑level workflow described in the
project specification and implements all five stages: input parsing,
graph construction, feature extraction, GAT modelling, and classification
with thresholding.  A separate neighbour‑sampling implementation is provided
in the `truthtrace_neighbor` directory.

## Overview

TruthTrace aims to detect misinformation by examining **how news propagates**
rather than relying solely on textual cues.  Each news item and its
propagation history are treated as a small graph; news nodes and user
nodes are connected by retweet, reply or follower edges.  A Graph
Attention Network (GAT) ingests both **content features** (text
embeddings) and **user features** and outputs a probability that the
news item is disinformation.  This codebase implements the high‑level
workflow proposed in the specification:

1. **Input** — Load a collection of news items and their associated
   propagation structures from the UPFD dataset.
2. **Graph construction** — Build a heterogeneous graph where nodes
   represent both news posts and users; edges encode retweets, replies
   and other interactions.
3. **Feature extraction** — Generate text embeddings for the news posts
   using a transformer model (e.g. BERT) and derive simple user‑level
   features (e.g. degree statistics).
4. **GNN model** — Apply a Graph Attention Network to aggregate
   neighbour information and produce graph‑level outputs (fake vs.
   real).
5. **Classifier & thresholding** — Train a classifier to output a
   probability of disinformation and compare it against a strong
   text‑only baseline.

An interactive Flask dashboard visualises each propagation graph and
displays predictions from both the baseline and GAT models.  Users can
select a news item, explore its retweet tree, and inspect the model
scores.

## Dataset

The **User Preference‑Aware Fake News Detection (UPFD)** dataset is used
for both training and evaluation.  The
code no longer relies on a bespoke JSON file; instead, it calls
`torch_geometric.datasets.UPFD`, which automatically downloads the
requested split and feature type.  Each call returns a list of
`Data` objects, where each object encodes one propagation cascade.

Available sub‑datasets: **`politifact`** and **`gossipcop`**.
Available feature types: **`bert`**, **`content`**, **`spacy`**, and
**`profile`**.  See the PyTorch Geometric documentation for details on
the feature dimensions.

When training or running the application you specify the root
directory for caching, the dataset name and the feature type.  If the
dataset files are not found locally, they will be downloaded
automatically.

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists versions of PyTorch and PyTorch Geometric
compatible with the code.  You will also need the `transformers` package
for text embeddings, `scikit-learn` for the baseline, and `flask` for
the web dashboard.

## Training

Run the training script from the repository root.  You must specify
three key parameters:

* `--root`: the directory where the UPFD dataset will be cached.
* `--dataset`: which sub‑dataset to use (`politifact` or `gossipcop`).
* `--feature`: which node feature type to load (`bert`, `content`, `spacy` or `profile`).

For example, to train on the Politifact portion using BERT embeddings and
save models to `models/`:

```bash
python train.py \
  --root ./data/upfd \
  --dataset politifact \
  --feature bert \
  --model_dir models/ \
  --epochs 20 \
  --hidden_dim 64 \
  --num_heads 4 \
  --batch_size 4
```

The script will automatically download the selected UPFD splits (train,
val, test) if they are not already present.  It trains two models:

* **Baseline (text‑only) classifier** — a logistic regression model on
  top of the root node embeddings.
* **GAT classifier** — a two‑layer Graph Attention Network followed by
  graph pooling and a linear classifier.

Model checkpoints are saved to `model_dir`.  Hyperparameters (hidden
dimension, number of attention heads, epochs, batch size) are all
configurable via flags.

## Running the Dashboard

After training, start the Flask application.  Set environment
variables to indicate where the dataset is cached, which subset and
feature type to use, and where the models are stored:

```bash
export TRUTHTRACE_ROOT=./data/upfd
export TRUTHTRACE_DATASET=politifact
export TRUTHTRACE_FEATURE=bert
export TRUTHTRACE_MODEL_DIR=models
python app.py
```

Then navigate to `http://127.0.0.1:5000` in your browser.  The main
page displays a dropdown of news IDs (corresponding to cascades in the
UPFD train/val/test splits).  Selecting one loads the
corresponding propagation graph, shows the predicted probability of
misinformation from both the baseline and GAT models, and colours
nodes according to their role.  The dashboard uses D3.js for
interactive graph visualisation.