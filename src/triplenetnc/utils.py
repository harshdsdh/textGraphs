import numpy as np
import torch
import networkx as nx
from torch.nn.utils.rnn import pad_sequence

model_dict = {
    "smallLM": "sentence-transformers/all-MiniLM-L12-v2",  # 22.7M params
    "mediumLM": "google/embeddinggemma-300m",
    "1B": "Qwen/Qwen3-Embedding-0.6B",  # 0.6B params
    "8B": "Qwen/Qwen3-Embedding-8B",  # 8B params
}


def original_and_duplicated_edges(edges, device):
    """return directed and undirected network"""
    edge_np = edges.astype(np.int64)
    edge_np = np.unique(edge_np, axis=1)

    reverse = edge_np[[1, 0], :]
    edge_np_rev = np.concatenate([edge_np, reverse], axis=1)
    edge_np_rev = np.unique(edge_np_rev, axis=1)
    edge_np_rev = edge_np_rev[:, edge_np_rev[0] != edge_np_rev[1]]

    edge_orig_index = torch.from_numpy(edge_np).long().contiguous().to(device)
    edge_rev_index = torch.from_numpy(edge_np_rev).long().contiguous().to(device)

    return edge_orig_index, edge_rev_index


# make semantic network


def add_edges_graph(
    vecs: list, original_text: list, train_mask, val_mask, test_mask
) -> nx.Graph:
    """create a semantic network"""
    N = len(vecs)
    batch_size = 5000
    K = 50  # changed here
    k = 25
    m = 0
    delta = 1.0

    ind = np.empty((N, K), np.int32)
    val = np.empty((N, K), np.float32)
    rho = np.empty(N, np.float32)

    vecs = np.array(vecs)
    for start_batch in range(0, N, batch_size):
        end_batch = min(N, start_batch + batch_size)
        X = vecs[start_batch:end_batch]

        S_cos = X @ vecs.T

        tau = 1.0 - S_cos

        r = np.arange(end_batch - start_batch)
        tau[r, start_batch + r] = np.inf
        m = max(m, np.max(tau[np.isfinite(tau)]))

        ind_temp = np.argpartition(tau, K - 1, axis=1)[:, :K]
        val_temp = np.take_along_axis(tau, ind_temp, axis=1)

        order = np.argsort(val_temp, axis=1)

        ind_temp = np.take_along_axis(ind_temp, order, axis=1)
        val_temp = np.take_along_axis(val_temp, order, axis=1)

        rho_temp = val_temp[:, k - 1]

        rho[start_batch:end_batch] = rho_temp
        ind[start_batch:end_batch] = ind_temp
        val[start_batch:end_batch] = val_temp

        print(f"batch: {start_batch}:{end_batch}")

    G = nx.Graph()
    for i in range(N):
        G.add_node(
            i,
            label=original_text[i],
            embs=vecs[i],
            train_mask=train_mask[i].item(),
            val_mask=val_mask[i].item(),
            test_mask=test_mask[i].item(),
        )
        for row_ind, index in enumerate(ind[i]):
            if val[i][row_ind] < delta * (np.sqrt(rho[i] * rho[index])):
                G.add_edge(i, int(index), weight=1.0 - (val[i][row_ind] / m))

    return G


def process_semantic_edges(graph, device):
    """process edges from semantic graph"""

    sem_edges = []
    edge_weights = []

    for u, v, attrs in graph.edges(data=True):

        w = attrs["weight"]

        sem_edges.append([u, v])
        sem_edges.append([v, u])

        edge_weights.append(w)
        edge_weights.append(w)

    edge_sem_index = (
        torch.tensor(sem_edges, dtype=torch.long).t().contiguous().to(device)
    )

    edge_weight = torch.tensor(edge_weights, dtype=torch.float).to(device)
    return edge_sem_index, edge_weight


def build_semantic_batch(edge_sem_index, edge_weight, x, max_neighbors=5):
    N, feat_dim = x.shape

    # edge_sem_index is already symmetric (both directions pushed in construction)
    src, dst = edge_sem_index[0].tolist(), edge_sem_index[1].tolist()
    w = edge_weight.tolist()

    neighbor_weight_lists = [[] for _ in range(N)]
    for s, d, wt in zip(src, dst, w):
        neighbor_weight_lists[s].append((d, wt))

    feats_list, lengths = [], []
    for i in range(N):
        pairs = neighbor_weight_lists[i]
        if len(pairs) == 0:
            feats_list.append(torch.empty(0, feat_dim, dtype=x.dtype))
            lengths.append(0)
            continue

        pairs.sort(key=lambda p: p[1], reverse=True)
        top = pairs[:max_neighbors]
        selected = torch.tensor([p[0] for p in top], dtype=torch.long)

        feats_list.append(x[selected])
        lengths.append(len(top))

    feats = pad_sequence(feats_list, batch_first=True, padding_value=0.0)
    max_len = max(feats.size(1), 1)
    pad_mask = torch.ones(N, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l > 0:
            pad_mask[i, :l] = False
    return feats, pad_mask
