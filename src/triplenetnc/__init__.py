import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import argparse
from pathlib import Path

from copy import deepcopy
from torch_geometric.utils import to_networkx
import networkx as nx
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence

from .utils import (
    model_dict,
    original_and_duplicated_edges,
    add_edges_graph,
    process_semantic_edges,
    build_semantic_batch,
)

from .models import SemanticModule, BiTopoModule, DirTopoModule, CompleteModel

### set device
if torch.cuda.is_available():
    print("Using GPU")
    print(torch.cuda.device_count())
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


def load_dataset(graph_name: str):
    """load dataset based on passed args"""
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "datasets" / "HeTBG" / f"{graph_name}.npz"
    # path = Path(f"./src/triplenetnc/datasets/HeTBG/{graph_name}.npz")
    if path.is_file():
        print(f"{graph_name} dataset available")

    dataset = np.load(path)
    return dataset


def is_symmetric(e):
    """check if graph is directed or undirected"""
    e_set = set(zip(e[0], e[1]))
    symm = all((v, u) in e_set for (u, v) in e_set)
    t = "directed" if symm is False else "undirected"
    print(f"graph type is: {t}")


def load_embs(graph_name, feat, raw_text):
    """load embedding given embedding model"""
    path = Path(f"./embs/{graph_name}/{feat}.pt")
    if path.is_file():
        print("saved emb found, loading saved embedding")
        print(path)
        embeddings = torch.load(path, weights_only=True)
    else:
        batch_size = 32
        encoder = SentenceTransformer(model_dict[feat], device=device)

        embeddings = encoder.encode(
            raw_text,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.cpu().float()
        del encoder

    return embeddings


def main() -> None:

    parser = argparse.ArgumentParser(description="run a gnn on a graph dataset")
    parser.add_argument("graph_name", help="name of a graph dataset, such as cora")
    args = parser.parse_args()
    graph_name = args.graph_name

    # load dataset
    dataset = load_dataset(graph_name)
    edge = dataset["edges"]
    labels = dataset["node_labels"]
    features = dataset["node_features"]
    train_mask = dataset["train_masks"]
    val_mask = dataset["val_masks"]
    test_mask = dataset["test_masks"]
    node_texts = dataset["node_texts"]

    is_symmetric(edge)

    for feats in [
        "BOW",
        "smallLM",
        "mediumLM",
        "1B",
        "8B",
    ]:
        print(f"LM used : {feats}")
        if feats == "BOW":
            features = dataset["node_features"]
        else:
            features = load_embs(graph_name, feats, node_texts)

        train = np.flatnonzero(train_mask)
        val = np.flatnonzero(val_mask)
        test = np.flatnonzero(test_mask)

        nclass = len(np.unique(labels))

        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)
        train = torch.LongTensor(train).to(device)
        val = torch.LongTensor(val).to(device)
        test = torch.LongTensor(test).to(device)

        edge_orig_index, edge_rev_index = original_and_duplicated_edges(edge, device)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        G_semantic = add_edges_graph(
            features_np, node_texts, train_mask, val_mask, test_mask
        )
        sem_edges, sem_edge_weights = process_semantic_edges(G_semantic, device)

        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        features_norm = features_np / (norms + 1e-8)
        features_norm = torch.as_tensor(features_norm, dtype=torch.float32)

        x = features_norm.to(device)
        bi_edge_index = edge_rev_index.to(device)
        dir_edge_index = edge_orig_index.to(device)
        labels = labels.to(device)

        train_mask = torch.as_tensor(train_mask, dtype=torch.bool).to(device)
        val_mask = torch.as_tensor(val_mask, dtype=torch.bool).to(device)
        test_mask = torch.as_tensor(test_mask, dtype=torch.bool).to(device)

        sem_feats, sem_pad_mask = build_semantic_batch(
            sem_edges.cpu(), sem_edge_weights.cpu(), x.cpu(), max_neighbors=25
        )

        sem_feats = sem_feats.to(device)
        sem_pad_mask = sem_pad_mask.to(device)

        epochs = 500
        patience = 50

        input_dim = features_norm.shape[1]

        all_results = []

        seeds = range(0, 100, 10)
        for heads in [4, 8, 16]:
            for lr in [1e-3]:
                for wd in [1e-3, 1e-6]:
                    for dropout in [0.3, 0.4]:
                        for hidden_dim in [128, 64]:
                            print(
                                f"testing config: head {heads}, lr {lr}, wd {wd}, dropout {dropout}, hidden_dim {hidden_dim}"
                            )
                            config_val_acc = []
                            config_test_acc = []
                            for seed in seeds:
                                random.seed(seed)
                                np.random.seed(seed)
                                torch.manual_seed(seed)

                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(seed)

                                print(f"  seed: {seed}", end=" ")

                                model = CompleteModel(
                                    feat_dim=input_dim,
                                    n_classes=nclass,
                                    model_dim=64,
                                    n_heads=heads,
                                    hidden_dim=hidden_dim,
                                    dropout=dropout,
                                ).to(device)

                                optimizer = torch.optim.Adam(
                                    model.parameters(), lr=lr, weight_decay=wd
                                )

                                best_val_acc = float("-inf")
                                best_state = None
                                epochs_without_improvement = 0
                                stopped_epochs = epochs

                                for epoch in range(epochs):
                                    model.train()
                                    optimizer.zero_grad(set_to_none=True)

                                    logits = model(
                                        x,
                                        sem_feats,
                                        sem_pad_mask,
                                        bi_edge_index,
                                        dir_edge_index,
                                    )
                                    loss = F.cross_entropy(
                                        logits[train_mask], labels[train_mask]
                                    )
                                    loss.backward()
                                    optimizer.step()

                                    model.eval()
                                    with torch.no_grad():
                                        logits = model(
                                            x,
                                            sem_feats,
                                            sem_pad_mask,
                                            bi_edge_index,
                                            dir_edge_index,
                                        )

                                        val_pred = logits[val_mask].argmax(dim=1)
                                        val_acc = (
                                            (val_pred == labels[val_mask])
                                            .float()
                                            .mean()
                                            .item()
                                        )
                                        if val_acc > best_val_acc:
                                            best_val_acc = val_acc
                                            best_state = deepcopy(model.state_dict())
                                            epochs_without_improvement = 0
                                        else:
                                            epochs_without_improvement += 1
                                        if epochs_without_improvement >= patience:
                                            print(f"early stopped: {epochs}")
                                            break

                                model.load_state_dict(best_state)
                                model.eval()
                                with torch.no_grad():
                                    logits = model(
                                        x,
                                        sem_feats,
                                        sem_pad_mask,
                                        bi_edge_index,
                                        dir_edge_index,
                                    )
                                    test_pred = logits[test_mask].argmax(dim=1)
                                    test_acc = (
                                        (test_pred == labels[test_mask])
                                        .float()
                                        .mean()
                                        .item()
                                    )
                                config_test_acc.append(test_acc)
                                config_val_acc.append(best_val_acc)
                                print(f"best val acc: {best_val_acc}")

                                all_results.append(
                                    {
                                        "seed": seed,
                                        "n_heads": heads,
                                        "lr": lr,
                                        "weight_decay": wd,
                                        "dropout": dropout,
                                        "hidden_dim": hidden_dim,
                                        "best_val_acc": best_val_acc,
                                        "test_acc": test_acc,
                                    }
                                )

                            mean_test = np.mean(config_test_acc)
                            std_test = np.mean(config_test_acc)

                            mean_val = np.mean(config_val_acc)
                            std_val = np.mean(config_val_acc)
                            print("config summary")
                            print(f"val acc:{mean_val}, std :{std_val}")
                            print(f"test acc:{mean_test}, std :{std_test}")

        results_df = pd.DataFrame(all_results)
        summary_df = (
            (
                results_df.groupby(
                    ["n_heads", "lr", "weight_decay", "dropout", "hidden_dim"]
                )
            )
            .agg(
                mean_val_acc=("best_val_acc", "mean"),
                std_val_acc=("best_val_acc", "std"),
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "mean"),
            )
            .reset_index()
        )

        summary_df = summary_df.sort_values(
            "mean_val_acc", ascending=False
        ).reset_index(drop=True)
        best_config = summary_df.iloc[0]

        print(f"type of language model used: {feats}")
        print(f"heads : {best_config['n_heads']}")
        print(f"learning rate: {best_config['lr']}")
        print(f"weight decay: {best_config['weight_decay']}")
        print(f"dropout : {best_config['dropout']}")
        print(f"hidden_dim: {best_config['hidden_dim']}")

        print(
            f"mean val acc and std : {best_config['mean_val_acc']}, {best_config['std_val_acc']}"
        )
        print(
            f"mean test acc and std : {best_config['mean_test_acc']}, {best_config['std_test_acc']}"
        )


if __name__ == "__main__":
    main()
