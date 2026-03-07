import os
import argparse
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.utils import to_undirected

from common import set_seed


def load_graph_dataset(dataset_name: str, device: str, re_split: bool = False, path_prefix: str = "."):
    try:
        graph_data = torch.load(f"{path_prefix}/datasets/{dataset_name}.pt", weights_only=False).to(device)
        graph_data.edge_index = to_undirected(graph_data.edge_index)
        return graph_data
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None


def load_predictions_and_labels(pred_file):
    predictions = []
    ground_truth = []
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Handle different JSON formats and potential errors
                    data = json.loads(line.strip())
                    
                    pred = None
                    for key in ['predict', 'prediction', 'output', 'generated_text']:
                        if key in data:
                            pred = data[key].strip()
                            break
                    
                    if pred is None:
                        for key, value in data.items():
                            if isinstance(value, str) and key != 'label':
                                pred = value.strip()
                                break

                    if pred is None:
                        continue
                    
                    label = None
                    if 'label' in data:
                        label = data['label'].strip()
                        for token in ['<|eot_id|>', '</s>', '<s>', '</end>']:
                            label = label.split(token)[0].strip()
                        
                        ground_truth.append(label)
                    
                    predictions.append(pred)
                except (json.JSONDecodeError, Exception):
                    continue
    except Exception as e:
        raise RuntimeError(f"Error reading prediction file: {e}")
    
    return predictions, ground_truth


def get_label_space(dataset_name, device, path_prefix="."):
    base_dataset = dataset_name.replace('_sft', '')
    graph_data = load_graph_dataset(base_dataset, device, path_prefix=path_prefix)
    
    if graph_data is None:
        print(f"Warning: Could not load dataset {base_dataset}. Using default label space.")
        return ["default_label"]
    
    return graph_data.label_name


def evaluate_predictions(predictions, ground_truth, label_space, topk=None):
    if not predictions or not ground_truth:
        print("Warning: No predictions or ground truth available for evaluation")
        return 0.0, 0.0, []
    
    if topk is not None:
        predictions = [pred.split(',')[0] for pred in predictions]
    
    total = len(predictions)
    correct = 0
    incorrect_cases = []

    label_to_id = {label.lower().strip(): i for i, label in enumerate(label_space)}
    pred_ids = []
    true_ids = []

    for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
        pred = pred.lower().strip()
        true = true.lower().strip()
        
        true_id = None
        for label in label_space:
            label_lower = label.lower()
            if true == label_lower or true in label_lower or label_lower in true:
                true_id = label_to_id[label_lower]
                break
        
        if true_id is None:
            true_id = 0
        
        if pred == true or pred in true or true in pred:
            correct += 1
            pred_ids.append(true_id)
        else:
            incorrect_cases.append({
                'index': i,
                'prediction': pred,
                'ground_truth': true
            })
            
            matched = False
            for label in label_space:
                label_lower = label.lower()
                if pred in label_lower or label_lower in pred:
                    pred_ids.append(label_to_id[label_lower])
                    matched = True
                    break
            
            if not matched:
                pred_ids.append(true_id)
        
        true_ids.append(true_id)

    accuracy = correct / total if total > 0 else 0
    f1 = f1_score(true_ids, pred_ids, average='macro')

    return accuracy, f1, incorrect_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", 
                        help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run evaluation on")
    parser.add_argument("--set_seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--topk", type=int, default=None,
                        help="Top-k predictions to consider (None for no splitting)")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to prediction file (.jsonl)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: directory of pred_file + '/evaluation')")
    parser.add_argument("--model_name", type=str, default="custom_model",
                        help="Name of model for result directory")
    parser.add_argument("--path_prefix", type=str, default=".",
                        help="Path prefix for dataset loading")
    parser.add_argument("--format_predictions", action="store_true",
                        help="Format predictions from raw data before evaluation")
    parser.add_argument("--raw_data", type=str, default=None,
                        help="Path to raw prediction data (when format_predictions is True)")

    args = parser.parse_args()

    if args.output_dir is None:
        pred_dir = os.path.dirname(args.pred_file)
        args.output_dir = os.path.join(pred_dir, "evaluation")
    
    eval_dir = os.path.join(args.output_dir, args.model_name)
    
    os.makedirs(eval_dir, exist_ok=True)

    set_seed(args.set_seed)
    
    if args.format_predictions and args.raw_data:
        format_predictions(args.raw_data, args.pred_file)

    predictions, ground_truth = load_predictions_and_labels(args.pred_file)

    if not ground_truth:
        metrics_file = os.path.join(eval_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                "dataset": args.dataset,
                "model": args.model_name,
                "error": "No labels found in prediction file"
            }, f, indent=2)
        print("Error: No labels found in prediction file")
        return

    label_space = get_label_space(args.dataset, args.device, path_prefix=args.path_prefix)

    accuracy, f1, incorrect_cases = evaluate_predictions(predictions, ground_truth, label_space, args.topk)

    print(f"[{args.dataset}] {args.model_name}: Acc={accuracy:.4f}, F1={f1:.4f} ({len(predictions)} samples)")

    incorrect_file = os.path.join(eval_dir, "incorrect_predictions.json")
    metrics_file = os.path.join(eval_dir, "metrics.json")

    with open(incorrect_file, 'w') as f:
        json.dump(incorrect_cases, f, indent=2)

    with open(metrics_file, 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model_name,
            "accuracy": accuracy,
            "macro_f1": f1,
            "total_samples": len(predictions),
            "incorrect_samples": len(incorrect_cases)
        }, f, indent=2)



def format_predictions(raw_data_path, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        formatted_data = []
        for line in lines:
            try:
                data = json.loads(line.strip())
                formatted_data.append(data)
            except json.JSONDecodeError:
                parts = line.strip().split('","')
                if len(parts) >= 2:
                    pred = parts[0].replace('{"predict": "', '').strip()
                    label = parts[1].replace('label": "', '').replace('</s>"}', '').strip()
                    formatted_data.append({"predict": pred, "label": label})
                else:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
                
        print(f"Formatted {len(formatted_data)} predictions to {output_path}")
    except Exception as e:
        print(f"Error formatting predictions: {e}")


if __name__ == "__main__":
    main()