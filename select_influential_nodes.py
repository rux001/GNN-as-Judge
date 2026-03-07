import argparse
import json
import os
import random
from multiprocessing import Pool
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from common import create_few_shot_dataset, load_graph_dataset, set_seed

SMALL_DATASETS = {"cora", "citeseer", "cornell", "wisconsin"}
DEFAULT_MAX_SUBGRAPH_NODES = 3000


def get_geometric_mean_degree(degrees: list) -> float:
    if not degrees or any(d <= 0 for d in degrees):
        return float('inf')
    return np.exp(np.mean(np.log(degrees)))


def extract_k_hop_subgraph(graph: nx.Graph, labeled_nodes: list, k_hops: int = 2) -> nx.Graph:
    subgraph_nodes = set(labeled_nodes)
    current_frontier = set(labeled_nodes)

    for hop in range(k_hops):
        next_frontier = set()
        for node in current_frontier:
            if node in graph:
                neighbors = set(graph.neighbors(node))
                next_frontier.update(neighbors)

        subgraph_nodes.update(next_frontier)
        current_frontier = next_frontier

    subgraph = graph.subgraph(subgraph_nodes).copy()
    return subgraph


def extract_adaptive_subgraph(graph: nx.Graph, seed_nodes: list, max_nodes: int = 3000) -> nx.Graph:
    subgraph_nodes = set(seed_nodes)
    current_frontier = set(seed_nodes)

    if len(subgraph_nodes) >= max_nodes:
        return graph.subgraph(subgraph_nodes).copy()

    hop = 0
    while True:
        hop += 1
        # Collect all new neighbors from the current frontier
        next_frontier = set()
        for node in current_frontier:
            if node in graph:
                for nb in graph.neighbors(node):
                    if nb not in subgraph_nodes:
                        next_frontier.update({nb})

        if len(next_frontier) == 0:
            # No more expansion possible
            break

        budget = max_nodes - len(subgraph_nodes)

        if len(next_frontier) <= budget:
            # Entire frontier fits within budget
            subgraph_nodes.update(next_frontier)
            current_frontier = next_frontier
            if len(subgraph_nodes) >= max_nodes:
                break
        else:
            # Sample from the frontier to stay near the target
            sampled = set(random.sample(sorted(next_frontier), budget))
            subgraph_nodes.update(sampled)
            break

    subgraph = graph.subgraph(subgraph_nodes).copy()
    return subgraph


def filter_connected_unlabeled_nodes(subgraph: nx.Graph, train_nodes: list, unlabeled_nodes: list, max_distance: int) -> list:
    subgraph_nodes = set(subgraph.nodes())
    unlabeled_in_subgraph = [node for node in unlabeled_nodes if node in subgraph_nodes]

    connected_unlabeled = set()
    train_nodes_in_subgraph = [node for node in train_nodes if node in subgraph_nodes]

    for train_node in train_nodes_in_subgraph:
        try:
            distances = nx.single_source_shortest_path_length(subgraph, train_node, cutoff=max_distance)

            for node, dist in distances.items():
                if node in unlabeled_in_subgraph and dist <= max_distance:
                    connected_unlabeled.add(node)
        except Exception:
            pass

    return list(connected_unlabeled)


def compute_paths_from_single_train_node_subgraph(args_tuple) -> Tuple[int, Dict, Dict]:
    t_node, candidate_nodes, subgraph_edges, max_distance = args_tuple

    G = nx.Graph()
    G.add_edges_from(subgraph_edges)

    distances = {}
    path_counts = {}

    try:
        single_source_distances = nx.single_source_shortest_path_length(G, t_node, cutoff=max_distance)

        for u_node in candidate_nodes:
            if u_node in single_source_distances:
                dist = single_source_distances[u_node]
                if dist <= max_distance:
                    distances[(t_node, u_node)] = dist

                    try:
                        paths = list(nx.all_shortest_paths(G, t_node, u_node))
                        path_counts[(t_node, u_node)] = min(len(paths), 1000)
                    except nx.NetworkXNoPath:
                        path_counts[(t_node, u_node)] = 0
    except Exception:
        pass

    return t_node, distances, path_counts


def precompute_shortest_path_info_subgraph(subgraph: nx.Graph, train_nodes: list, candidate_nodes: list, max_distance: int) -> Tuple[Dict, Dict]:
    distances = {}
    path_counts = {}

    subgraph_train_nodes = [node for node in train_nodes if node in subgraph]
    subgraph_edges = list(subgraph.edges())

    args_list = [
        (t_node, candidate_nodes, subgraph_edges, max_distance)
        for t_node in subgraph_train_nodes
    ]

    with Pool() as pool:
        results = list(tqdm(
            pool.imap(compute_paths_from_single_train_node_subgraph, args_list),
            total=len(subgraph_train_nodes),
            desc="Computing paths (subgraph)"
        ))

    for t_node, node_distances, node_path_counts in results:
        distances.update(node_distances)
        path_counts.update(node_path_counts)

    return distances, path_counts


def compute_influence_for_candidate_node(args_tuple) -> Tuple[int, float]:
    u_node, train_nodes, distances, path_counts, node_degrees, subgraph_edges = args_tuple

    G = nx.Graph()
    G.add_edges_from(subgraph_edges)

    max_influence = 0.0

    for t_node in train_nodes:
        key = (t_node, u_node)

        if key not in distances:
            continue

        h_star = distances[key]
        if h_star == 0:
            continue

        num_paths = path_counts[key]
        if num_paths == 0:
            continue

        min_geo_mean = calculate_min_geometric_mean_degree(G, t_node, u_node, node_degrees)

        if min_geo_mean == float('inf') or min_geo_mean == 0:
            continue

        influence = num_paths / (min_geo_mean ** h_star)
        max_influence = max(max_influence, influence)

    return u_node, max_influence


def calculate_influence_scores_subgraph_connected(graph: nx.Graph, train_nodes: list, unlabeled_nodes: list,
                                                k_hops: int = 2, max_distance: int = 4) -> Dict[int, float]:
    subgraph = extract_k_hop_subgraph(graph, train_nodes, k_hops=k_hops)

    connected_nodes = filter_connected_unlabeled_nodes(subgraph, train_nodes, unlabeled_nodes, max_distance)

    if len(connected_nodes) == 0:
        return {}

    node_degrees = dict(subgraph.degree())
    distances, path_counts = precompute_shortest_path_info_subgraph(
        subgraph, train_nodes, connected_nodes, max_distance
    )

    subgraph_edges = list(subgraph.edges())
    train_nodes_in_subgraph = [node for node in train_nodes if node in subgraph]

    args_list = [
        (u_node, train_nodes_in_subgraph, distances, path_counts, node_degrees, subgraph_edges)
        for u_node in connected_nodes
    ]

    with Pool() as pool:
        results = list(tqdm(
            pool.imap(compute_influence_for_candidate_node, args_list),
            total=len(connected_nodes),
            desc="Computing influence (subgraph)"
        ))

    influence_scores = dict(results)
    return influence_scores


def calculate_influence_scores_adaptive(graph: nx.Graph, train_nodes: list, unlabeled_nodes: list,
                                        max_subgraph_nodes: int = 3000, max_distance: int = 4) -> Dict[int, float]:
    """Influence scoring on an adaptively-extracted subgraph.
    Grows the subgraph around labeled nodes up to max_subgraph_nodes, then
    runs the standard influence computation on this manageable subgraph.
    """
    subgraph = extract_adaptive_subgraph(graph, train_nodes, max_nodes=max_subgraph_nodes)

    connected_nodes = filter_connected_unlabeled_nodes(subgraph, train_nodes, unlabeled_nodes, max_distance)

    if len(connected_nodes) == 0:
        return {}

    node_degrees = dict(subgraph.degree())
    distances, path_counts = precompute_shortest_path_info_subgraph(
        subgraph, train_nodes, connected_nodes, max_distance
    )

    subgraph_edges = list(subgraph.edges())
    train_nodes_in_subgraph = [node for node in train_nodes if node in subgraph]

    args_list = [
        (u_node, train_nodes_in_subgraph, distances, path_counts, node_degrees, subgraph_edges)
        for u_node in connected_nodes
    ]

    with Pool() as pool:
        results = list(tqdm(
            pool.imap(compute_influence_for_candidate_node, args_list),
            total=len(connected_nodes),
            desc="Computing influence (adaptive)"
        ))

    influence_scores = dict(results)
    return influence_scores


def calculate_min_geometric_mean_degree(graph: nx.Graph, source: int, target: int, node_degrees: Dict) -> float:
    min_geo_mean = float('inf')

    try:
        for path in nx.all_shortest_paths(graph, source, target):
            degrees = [node_degrees[node] for node in path]
            geo_mean = get_geometric_mean_degree(degrees)
            min_geo_mean = min(min_geo_mean, geo_mean)
    except nx.NetworkXNoPath:
        return float('inf')

    return min_geo_mean


def compute_paths_from_single_train_node_full(args_tuple) -> Tuple[int, Dict, Dict]:
    t_node, unlabeled_nodes, graph_edges = args_tuple

    G = nx.Graph()
    G.add_edges_from(graph_edges)

    distances = {}
    path_counts = {}

    try:
        single_source_distances = nx.single_source_shortest_path_length(G, t_node)

        for u_node in unlabeled_nodes:
            if u_node in single_source_distances:
                dist = single_source_distances[u_node]
                distances[(t_node, u_node)] = dist

                try:
                    path_count = len(list(nx.all_shortest_paths(G, t_node, u_node)))
                    path_counts[(t_node, u_node)] = path_count
                except nx.NetworkXNoPath:
                    path_counts[(t_node, u_node)] = 0
    except Exception:
        pass

    return t_node, distances, path_counts


def precompute_shortest_path_info_full(graph: nx.Graph, train_nodes: list, unlabeled_nodes: list) -> Tuple[Dict, Dict]:
    distances = {}
    path_counts = {}

    graph_edges = list(graph.edges())

    args_list = [
        (t_node, unlabeled_nodes, graph_edges)
        for t_node in train_nodes
    ]

    with Pool() as pool:
        results = list(tqdm(
            pool.imap(compute_paths_from_single_train_node_full, args_list),
            total=len(train_nodes),
            desc="Computing paths (full)"
        ))

    for t_node, node_distances, node_path_counts in results:
        distances.update(node_distances)
        path_counts.update(node_path_counts)

    return distances, path_counts


def compute_influence_for_unlabeled_node_full(args_tuple) -> Tuple[int, float]:
    u_node, train_nodes, distances, path_counts, node_degrees, graph_edges = args_tuple

    G = nx.Graph()
    G.add_edges_from(graph_edges)

    max_influence = 0.0

    for t_node in train_nodes:
        key = (t_node, u_node)

        if key not in distances:
            continue

        h_star = distances[key]
        if h_star == 0:
            continue

        num_paths = path_counts[key]
        if num_paths == 0:
            continue

        min_geo_mean = calculate_min_geometric_mean_degree(G, t_node, u_node, node_degrees)

        if min_geo_mean == float('inf') or min_geo_mean == 0:
            continue

        influence = num_paths / (min_geo_mean ** h_star)
        max_influence = max(max_influence, influence)

    return u_node, max_influence


def calculate_influence_scores_full_graph(graph: nx.Graph, train_nodes: list, unlabeled_nodes: list) -> Dict[int, float]:
    node_degrees = dict(graph.degree())

    distances, path_counts = precompute_shortest_path_info_full(graph, train_nodes, unlabeled_nodes)

    graph_edges = list(graph.edges())

    args_list = [
        (u_node, train_nodes, distances, path_counts, node_degrees, graph_edges)
        for u_node in unlabeled_nodes
    ]

    with Pool() as pool:
        results = list(tqdm(
            pool.imap(compute_influence_for_unlabeled_node_full, args_list),
            total=len(unlabeled_nodes),
            desc="Computing influence (full)"
        ))

    influence_scores = dict(results)
    return influence_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path_prefix", type=str, default=".")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "subgraph", "full", "adaptive"],
                        help="Node selection method. 'auto' uses full graph for small "
                             "datasets (cora, citeseer) and adaptive subgraph for larger ones.")
    parser.add_argument("--k_hops", type=int, default=2,
                        help="Number of hops for subgraph extraction (subgraph method only)")
    parser.add_argument("--max_distance", type=int, default=4,
                        help="Maximum shortest-path distance for influence computation")
    parser.add_argument("--max_subgraph_nodes", type=int, default=DEFAULT_MAX_SUBGRAPH_NODES,
                        help="Target subgraph size for adaptive/auto method on large graphs")
    args = parser.parse_args()

    set_seed(args.seed)
    # Create split on GPU for consistency with other pipeline stages, then move to CPU for NetworkX
    split_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.shots:
        graph_data = create_few_shot_dataset(args.dataset, shots=args.shots, seed=args.seed, device=split_device, path_prefix=args.path_prefix)
    else:
        graph_data, _, _ = load_graph_dataset(args.dataset, device=split_device, path_prefix=args.path_prefix)

    graph_data = graph_data.to("cpu")
    edge_index = graph_data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))
    G.add_edges_from(edge_index.T)

    train_nodes = graph_data.train_mask.nonzero().squeeze().cpu().tolist()
    if isinstance(train_nodes, int):
        train_nodes = [train_nodes]
    unlabeled_nodes = graph_data.test_mask.nonzero().squeeze().cpu().tolist()
    if isinstance(unlabeled_nodes, int):
        unlabeled_nodes = [unlabeled_nodes]

    # Resolve 'auto' method
    method = args.method
    if method == "auto":
        if args.dataset.lower() in SMALL_DATASETS:
            method = "full"
        else:
            method = "adaptive"

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | "
          f"Train: {len(train_nodes)}, Unlabeled: {len(unlabeled_nodes)} | Method: {method}")

    if method == "full":
        scores = calculate_influence_scores_full_graph(G, train_nodes, unlabeled_nodes)
    elif method == "subgraph":
        scores = calculate_influence_scores_subgraph_connected(
            G, train_nodes, unlabeled_nodes,
            k_hops=args.k_hops,
            max_distance=args.max_distance
        )
    elif method == "adaptive":
        scores = calculate_influence_scores_adaptive(
            G, train_nodes, unlabeled_nodes,
            max_subgraph_nodes=args.max_subgraph_nodes,
            max_distance=args.max_distance
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    non_zero_scores = {k: v for k, v in scores.items() if v > 0}
    available_nodes = min(args.k, len(non_zero_scores))

    sorted_nodes = sorted(non_zero_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_nodes = [int(node_id) for node_id, score in sorted_nodes[:available_nodes]]

    print(f"Selected {len(top_k_nodes)}/{args.k} influential nodes "
          f"(from {len(non_zero_scores)} with non-zero influence)")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    metadata = {
        "selected_node_ids": top_k_nodes,
        "method": method,
        "total_influential": len(non_zero_scores),
        "total_unlabeled": len(unlabeled_nodes),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges()
    }

    if method == "subgraph":
        metadata.update({
            "k_hops": args.k_hops,
            "max_distance": args.max_distance
        })
    elif method == "adaptive":
        metadata.update({
            "max_subgraph_nodes": args.max_subgraph_nodes,
            "max_distance": args.max_distance
        })

    with open(args.output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
