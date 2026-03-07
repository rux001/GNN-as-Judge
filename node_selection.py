import torch
import torch.nn.functional as F
import json
from typing import Dict, Tuple, Optional, Any

def extract_category(prediction: str) -> str:
    return prediction.strip()

def load_llm_predictions(llm_predictions_file: str, graph_data) -> Dict[int, int]:
    label_map = {label.lower().strip(): i for i, label in enumerate(graph_data.label_name)}
    test_indices = torch.where(graph_data.test_mask)[0].cpu().numpy()
    
    raw_predictions = []
    try:
        with open(llm_predictions_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    pred_text = data.get('predict', data.get('prediction', data.get('answer', data.get('label', None))))
                    if isinstance(pred_text, str):
                        category = extract_category(pred_text.strip())
                        raw_predictions.append(category)
                except Exception:
                    pass
    except FileNotFoundError:
        return {}

    llm_predictions_mapped = {}
    n_predictions_to_map = min(len(raw_predictions), len(test_indices))
    
    for i in range(n_predictions_to_map):
        category = raw_predictions[i]
        node_id = int(test_indices[i])
        category_lower = category.lower().strip()
        
        if category_lower in label_map:
            llm_predictions_mapped[node_id] = label_map[category_lower]
        else:
            matched = False
            for label, idx in label_map.items():
                if category_lower in label or label in category_lower:
                    llm_predictions_mapped[node_id] = idx
                    matched = True
                    break
    
    return llm_predictions_mapped

def find_agreed_and_disagreed_nodes(gnn_predictions, llm_predictions, graph_data):
    agreed_nodes = {}
    disagreed_nodes = {}

    gnn_probs = F.softmax(gnn_predictions, dim=1)
    gnn_preds = gnn_predictions.argmax(dim=1)

    num_gnn_nodes = gnn_predictions.size(0)

    for node_idx, llm_pred_idx in llm_predictions.items():
        if not (0 <= node_idx < num_gnn_nodes):
            continue

        gnn_pred_idx = gnn_preds[node_idx].item()
        gnn_conf = gnn_probs[node_idx, gnn_pred_idx].item()

        if gnn_pred_idx == llm_pred_idx:
            agreed_nodes[node_idx] = (gnn_conf,)
        else:
            disagreed_nodes[node_idx] = (gnn_pred_idx, llm_pred_idx, gnn_conf)

    return agreed_nodes, disagreed_nodes


def compute_final_label_accuracy(final_selection, original_labels):
    correct = 0
    total = 0
    
    for node_id_str, label in final_selection["all_nodes"].items():
        node_idx = int(node_id_str)
        true_label = original_labels[node_idx].item()
        if label == true_label:
            correct += 1
        total += 1
    
    overall_accuracy = correct / total if total > 0 else 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'num_nodes': total,
        'correct': correct
    }

def load_llm_predictions_for_selected(llm_predictions_file: str, selected_node_ids: list, graph_data) -> Dict[int, int]:
    label_map = {label.lower().strip(): i for i, label in enumerate(graph_data.label_name)}

    def _normalize(text: str) -> str:
        return text.lower().strip()

    mapped: Dict[int, int] = {}
    raw = []
    with open(llm_predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                raw.append(json.loads(line))
            except Exception:
                continue

    n = min(len(raw), len(selected_node_ids))
    for i in range(n):
        node_id = int(selected_node_ids[i])
        rec = raw[i]
        pred_text = rec.get('predict', rec.get('prediction', rec.get('answer', rec.get('label', ''))))
        if not isinstance(pred_text, str):
            continue
        key = _normalize(pred_text)
        if key in label_map:
            mapped[node_id] = label_map[key]
            continue
        matched = False
        for lbl, idx in label_map.items():
            if key in lbl or lbl in key:
                mapped[node_id] = idx
                matched = True
                break

    return mapped

def filter_disagreed_by_preference(
    disagreed_nodes: Dict[int, Any],
    gnn_predictions: torch.Tensor,
    confidence_threshold: float
) -> Dict[int, Any]:
    filtered_disagreed = {}
    gnn_probs = F.softmax(gnn_predictions, dim=1)

    for node_id, triple in disagreed_nodes.items():
        if isinstance(triple, (list, tuple)) and len(triple) >= 2:
            gnn_pred_idx, llm_pred_idx = int(triple[0]), int(triple[1])
            gnn_conf = float(triple[2]) if len(triple) > 2 else float(gnn_probs[node_id, gnn_pred_idx].item())
        elif isinstance(triple, dict):
            gnn_pred_idx = int(triple.get('gnn_pred'))
            llm_pred_idx = int(triple.get('llm_pred'))
            gnn_conf = float(triple.get('gnn_conf', gnn_probs[node_id, gnn_pred_idx].item()))
        else:
            continue

        prob_gnn_class = gnn_probs[node_id, gnn_pred_idx].item()
        prob_llm_class = gnn_probs[node_id, llm_pred_idx].item()
        preference_score = prob_gnn_class - prob_llm_class
        if preference_score >= confidence_threshold:
            filtered_disagreed[node_id] = (gnn_pred_idx, llm_pred_idx, gnn_conf)
    return filtered_disagreed

def retrain_gnn_on_agreed(gnn_model: torch.nn.Module, graph_data, agreed_nodes: Dict[int, Any], device: str, lr: float = 1e-3, epochs: int = 100):
    gnn_model.train()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)
    train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
    for nid in agreed_nodes.keys():
        train_mask[nid] = True
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = gnn_model(graph_data.x, graph_data.edge_index)
        loss = F.cross_entropy(logits[train_mask], graph_data.y[train_mask]) if train_mask.any() else torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()
    gnn_model.eval()
    return gnn_model