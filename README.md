# TruthTrace Full Version (GAT on Full Propagation Trees)

This repository contains a complete implementation of **TruthTrace** for the
“full-batch” setting. It follows the high-level workflow described in the
project specification and implements all five stages: input parsing,
graph construction, feature extraction, GAT modelling, and classification
with thresholding. A separate neighbour-sampling implementation is provided
in the `truthtrace_neighbor` directory.

## Overview

TruthTrace aims to detect misinformation by examining **how news propagates**
rather than relying solely on textual cues. Each news item and its
propagation history are treated as a small graph; news nodes and user
nodes are connected by retweet, reply or follower edges. A Graph
Attention Network (GAT) ingests both **content features** (text
embeddings) and **user features** and outputs a probability that the
news item is disinformation, consistent with the original UPFD
benchmark design (Dou et al., 2021).

High-level workflow:

1. **Input** — Load a collection of news items and their associated
   propagation structures from the UPFD dataset.
2. **Graph construction** — Build a heterogeneous graph where nodes
   represent both news posts and users; edges encode retweets, replies
   and other interactions.
3. **Feature extraction** — Use pre-computed text embeddings
   (e.g. BERT-based features) and simple user-level features
   (e.g. degree statistics, profile features if available).
4. **GNN model (GAT)** — Apply a Graph Attention Network (Veličković
   et al., 2018) to aggregate neighbourhood information and produce
   graph-level outputs (fake vs. real).
5. **Classifier & thresholding** — Train a classifier to output a
   probability of disinformation and compare it against a strong
   text-only baseline (Devlin et al., 2019; Vosoughi et al., 2018).

An interactive Flask dashboard visualises each propagation graph and
displays predictions from both the baseline and GAT models. Users can
select a news item, explore its retweet tree, and inspect the model
scores.

## Dataset (UPFD from OpenDataLab)

This project uses the **User Preference-Aware Fake News Detection
(UPFD)** dataset (Dou et al., 2021), but **does not rely on
`torch_geometric.datasets.UPFD` to download it**. Instead, you download
the data manually from **OpenDataLab** and place it into the expected
folder structure.

### 1. Download from OpenDataLab

1. Go to the UPFD page on OpenDataLab: https://opendatalab.com/OpenDataLab/UPFD/tree/main/raw.
2. Log in / create an account if needed.
3. Download the archive(s) containing the UPFD data.  

### 2. Create the local folder structure

Inside your project directory, create the following structure:

```text
truthtrace_full/
  data/
    politifact/
      raw/
        ... all Politifact UPFD files here ...
    gossipcop/
      raw/
        ... all GossipCop UPFD files here ...
