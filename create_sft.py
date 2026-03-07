import json
import torch
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from common import DIRECT_PROMPTS
from common.dataloader import create_few_shot_dataset

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_test_samples', type=int, default=1000)
    parser.add_argument('--path_prefix', type=str, default=".")
    
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return args

def create_sft_conversation(text: str, prompt: str, label: str) -> Dict[str, List[Dict[str, str]]]:
    query_content = f"{text}\n{prompt}"
    return {
        "conversations": [
            {"from": "human", "value": query_content},
            {"from": "gpt", "value": label}
        ]
    }

def create_unlabeled_dataset_from_test_mask(graph_data, dataset_name: str, output_path: str, suffix: str = "") -> Tuple[List[Dict], Optional[str]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")
    
    all_test_indices = torch.where(graph_data.test_mask)[0].cpu().tolist()
    
    indices_to_use = all_test_indices
    
    node_ids_dir = os.path.dirname(output_path)
    os.makedirs(node_ids_dir, exist_ok=True)
    node_ids_filename = f"{dataset_name}{suffix}_unlabeled_node_ids.json"
    node_ids_path = os.path.join(node_ids_dir, node_ids_filename)
    
    with open(node_ids_path, 'w') as f:
        json.dump({"selected_node_ids": indices_to_use}, f, indent=2)
    print(f"Saved {len(indices_to_use)} unlabeled node IDs to {node_ids_path}")
    
    dataset = []
    for idx in indices_to_use:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]          
        conversation = create_sft_conversation(origin_txt, prompt, true_label)
        dataset.append(conversation)
    
    return dataset, node_ids_path


def create_train_dataset(graph_data, dataset_name: str) -> List[Dict[str, Any]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")   
    
    dataset = []

    train_indices = torch.where(graph_data.train_mask)[0].cpu().tolist()
    for idx in train_indices:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]  
        conversation = create_sft_conversation(origin_txt, prompt, true_label)
        dataset.append(conversation)

    return dataset

def create_validation_dataset(graph_data, dataset_name: str) -> List[Dict[str, Any]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")

    dataset = []
    val_indices = torch.where(graph_data.val_mask)[0].cpu().tolist()

    for idx in val_indices:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]

        conversation = create_sft_conversation(origin_txt, prompt, true_label)
        dataset.append(conversation)

    return dataset

def create_test_dataset(graph_data, dataset_name: str, max_samples: int = -1) -> List[Dict[str, Any]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")

    dataset = []
    test_indices = torch.where(graph_data.test_mask)[0].cpu().tolist()

    if max_samples > 0 and len(test_indices) > max_samples:
        test_indices = random.sample(test_indices, max_samples)
        
    for idx in test_indices:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]
        conversation = create_sft_conversation(origin_txt, prompt, true_label)
        dataset.append(conversation)

    return dataset

def convert_to_sft_format(
    graph_data, 
    dataset_name: str, 
    output_path: str, 
    suffix: str = "",
    max_test_samples: int = -1,
):
    try:
        if suffix and not suffix.startswith('_'):
            suffix = f"_{suffix}"
            
        train_dataset = create_train_dataset(
            graph_data=graph_data,
            dataset_name=dataset_name
        )

        train_output_path = output_path.replace('.json', f'{suffix}_train.json')
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, ensure_ascii=False, indent=2)

        val_dataset = create_validation_dataset(
            graph_data=graph_data,
            dataset_name=dataset_name
        )

        if val_dataset:
            val_output_path = output_path.replace('.json', f'{suffix}_val.json')
            with open(val_output_path, 'w', encoding='utf-8') as f:
                json.dump(val_dataset, f, ensure_ascii=False, indent=2)

        test_dataset = create_test_dataset(
            graph_data=graph_data,
            dataset_name=dataset_name,
            max_samples=max_test_samples
        )

        test_output_path = output_path.replace('.json', f'{suffix}_test.json')
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, ensure_ascii=False, indent=2)

        unlabeled_dataset, node_ids_path = create_unlabeled_dataset_from_test_mask(
            graph_data=graph_data,
            dataset_name=dataset_name,
            output_path=output_path,
            suffix=suffix,
        )
        unlabeled_output_path = output_path.replace('.json', f'{suffix}_unlabeled.json')
        with open(unlabeled_output_path, 'w', encoding='utf-8') as f:
            json.dump(unlabeled_dataset, f, ensure_ascii=False, indent=2)

        print(f"SFT splits: train={len(train_dataset)}, val={len(val_dataset)}, "
              f"test={len(test_dataset)}, unlabeled={len(unlabeled_dataset)}")

    except Exception as e:
        print(f"Error creating SFT datasets: {str(e)}")
        raise

def main():
    try:
        args = parse_arguments()
        seed = args.seed

        graph_data = create_few_shot_dataset(
            args.dataset,
            args.shots,
            args.seed,
            args.device,
            path_prefix=args.path_prefix
        )
        suffix = f"_{args.shots}_shot"

        convert_to_sft_format(
            graph_data=graph_data,
            dataset_name=args.dataset,
            output_path=args.output,
            suffix=suffix,
            max_test_samples=args.max_test_samples,
        )

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())