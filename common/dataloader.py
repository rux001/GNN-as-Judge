import os
import torch
import numpy as np
from typing import Tuple
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from collections import defaultdict
import json
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def re_split_data(num_node, train_percent=0.6, val_percent=0.2, test_percent=0.2, device="cuda:0", seed=42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    assert abs(train_percent + val_percent + test_percent - 1.0) < 1e-6, "Percentages must sum to 1.0"
    
    node_ids = np.arange(num_node)
    np.random.shuffle(node_ids)
    
    train_ids = np.sort(node_ids[:int(num_node * train_percent)])
    val_ids = np.sort(node_ids[int(num_node * train_percent): int(num_node * (train_percent + val_percent))])
    test_ids = np.sort(node_ids[int(num_node * (train_percent + val_percent)):])
    
    train_mask = torch.zeros(num_node, dtype=torch.bool)
    val_mask = torch.zeros(num_node, dtype=torch.bool)
    test_mask = torch.zeros(num_node, dtype=torch.bool)
    
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def create_few_shot_dataset(dataset_name, shots=5, seed=42, device="cuda:0", path_prefix="."):
    """Create a few-shot dataset with deterministic splits.

    Always resets RNG to `seed` before sampling to guarantee identical splits
    across all pipeline stages regardless of prior RNG state.
    """
    import random as _random
    # Reset all RNGs to ensure identical splits across pipeline stages
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)

    graph_data = torch.load(f"{path_prefix}/datasets/{dataset_name}.pt", weights_only=False)
    graph_data.edge_index = to_undirected(graph_data.edge_index)
    graph_data = graph_data.to(device)

    num_classes = graph_data.y.max().item() + 1
    new_train_mask = torch.zeros_like(graph_data.train_mask)

    if dataset_name == "arxiv":
        ogb_dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
        split_idx = ogb_dataset.get_idx_split()

        official_train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
        official_val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
        official_test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)

        official_train_mask[split_idx['train'].to(device)] = True
        official_val_mask[split_idx['valid'].to(device)] = True
        official_test_mask[split_idx['test'].to(device)] = True

        for c in range(num_classes):
            class_mask = (graph_data.y == c) & official_train_mask
            class_indices = class_mask.nonzero().squeeze(-1)
            num_available = class_indices.shape[0]
            if num_available > 0:
                actual_shots = min(shots, num_available)
                selected_indices = torch.randperm(num_available, device=device)[:actual_shots]
                selected = class_indices[selected_indices]
                new_train_mask[selected] = True

        unused_train_nodes = official_train_mask & (~new_train_mask)
        new_test_mask = official_test_mask | unused_train_nodes

        graph_data.train_mask = new_train_mask
        graph_data.val_mask = official_val_mask
        graph_data.test_mask = new_test_mask

    else:
        for c in range(num_classes):
            class_indices = ((graph_data.y == c) & graph_data.train_mask).nonzero().squeeze(-1)
            num_available = class_indices.numel()
            actual_shots = min(shots, num_available)
            perm = torch.randperm(num_available, device=device)
            selected = class_indices[perm[:actual_shots]]
            new_train_mask[selected] = True

        val_indices = (~graph_data.train_mask).nonzero().squeeze(-1)
        perm = torch.randperm(val_indices.numel(), device=device)
        selected = val_indices[perm[:500]]
        new_val_mask = torch.zeros_like(graph_data.val_mask)
        new_val_mask[selected] = True

        graph_data.train_mask = new_train_mask
        graph_data.val_mask = new_val_mask
        graph_data.test_mask = ~(graph_data.train_mask | graph_data.val_mask)

    # Save/verify split IDs for consistency across pipeline stages
    import hashlib
    train_ids = graph_data.train_mask.nonzero().squeeze(-1).cpu().tolist()
    val_ids = graph_data.val_mask.nonzero().squeeze(-1).cpu().tolist()
    test_ids = graph_data.test_mask.nonzero().squeeze(-1).cpu().tolist()
    split_hash = hashlib.md5(str(train_ids + val_ids + test_ids).encode()).hexdigest()[:8]

    split_dir = os.path.join(path_prefix, "datasets", "splits")
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, f"{dataset_name}_{shots}shot_seed{seed}.json")

    if os.path.exists(split_file):
        import json as _json
        with open(split_file) as f:
            saved = _json.load(f)
        if saved["hash"] != split_hash:
            raise RuntimeError(
                f"Split mismatch for {dataset_name} {shots}-shot seed={seed}! "
                f"Saved hash={saved['hash']}, current hash={split_hash}. "
                f"Delete {split_file} to regenerate."
            )
    else:
        import json as _json
        with open(split_file, 'w') as f:
            _json.dump({
                "hash": split_hash,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_count": len(test_ids),
                "device": str(device),
            }, f, indent=2)

    print(f"[{dataset_name}] {shots}-shot: train={len(train_ids)}, "
          f"val={len(val_ids)}, test={len(test_ids)} (split_hash={split_hash})")
    return graph_data.to(device)


def load_graph_dataset(dataset_name: str, device: str, path_prefix: str = "."):
    file_path = f"{path_prefix}/datasets/{dataset_name}.pt"
    try:
        graph_data = torch.load(file_path, weights_only=False)
        graph_data.edge_index = to_undirected(graph_data.edge_index)
        return graph_data.to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset {file_path}: {str(e)}")

def load_graph_dataset_re_split(
    dataset_name: str, 
    device: str, 
    path_prefix: str = ".", 
    train_percent=0.6, 
    val_percent=0.2, 
    test_percent=None,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    graph_data = load_graph_dataset(dataset_name, "cpu", path_prefix)

    if test_percent is None:
        test_percent = 1.0 - train_percent - val_percent

    graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = re_split_data(
        graph_data.num_nodes, train_percent, val_percent, test_percent, "cpu", seed
    )
    return graph_data.to(device)

def load_graph_dataset_for_gnn(dataset_name, device, re_split=False, shots=20, path_prefix="../..", emb_model="shallow", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if re_split:
        graph_data = load_graph_dataset_re_split(dataset_name, device, path_prefix=path_prefix, seed=seed)
    else:
        graph_data = load_graph_dataset(dataset_name, device, path_prefix)

    if emb_model != "shallow":
        assert os.path.exists(f"{path_prefix}/datasets/{emb_model}/{dataset_name}.pt")
        node_feat = torch.load(f"{path_prefix}/datasets/{emb_model}/{dataset_name}.pt", map_location=device, weights_only=False).to(device).type(torch.float)
        graph_data.x = node_feat
    
    if emb_model == "shallow" and dataset_name in ["reddit", "instagram", "computer", "photo", "history", "cornell"]:
        if os.path.exists(f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt"):
            node_feat = torch.load(f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt", map_location=device).to(device)
        else:
            from node2vec import Node2Vec
            from torch_geometric.utils.convert import to_networkx
        
            nx_graph = to_networkx(graph_data)
            node2vec = Node2Vec(nx_graph, dimensions=300, walk_length=30, num_walks=10, workers=4)
            node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
            print(node2vec_model.wv.vectors.shape, type(node2vec_model.wv.vectors))
            node_feat = torch.FloatTensor(node2vec_model.wv.vectors).to(device)
            os.makedirs(f"{path_prefix}/datasets/Node2Vec", exist_ok=True)
            torch.save(node_feat, f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt")
        graph_data.x = node_feat
    
    return graph_data
