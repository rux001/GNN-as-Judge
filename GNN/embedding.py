"""
Optional utility for generating node embeddings using language models.

This script generates node embeddings from raw text using various text encoders
(SentenceBert, RoBERTa, Mistral, etc.) and saves them for use as GNN input features.

This is NOT required for the main GNN-as-Judge pipeline if you already have
pre-computed dataset .pt files with node features.

Requirements:
    pip install sentence-transformers transformers
"""

import torch
from tqdm import tqdm
import os
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import get_cur_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora",
                        choices=['cora', "pubmed", "citeseer", "wikics", "arxiv",
                                 "instagram", "reddit", "computer", "photo", "history",
                                 "ogbn-products_subset"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder_name", type=str, default="SentenceBert",
                        choices=["MiniLM", "SentenceBert", "e5-large", "roberta",
                                 "Qwen-3B", "Qwen-7B", "Mistral-7B", "Llama-8B"])
    parser.add_argument("--use_cls", type=int, default=1)
    parser.add_argument("--save_emb", type=int, default=1)

    args = parser.parse_args()

    device = torch.device(args.device)
    graph_data = torch.load(f"../datasets/{args.dataset}.pt", weights_only=False)

    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    if os.path.exists(f"../datasets/{args.encoder_name}/{args.dataset}.pt"):
        print(f"[{args.dataset}-{args.encoder_name}] Embedding file already exists, Quit!")
        print('= ' * 20)
        exit()

    # Import TextEncoder - requires sentence-transformers or transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Please install sentence-transformers: pip install sentence-transformers")
        exit(1)

    encoder_type = "LM" if args.encoder_name in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"

    # Use SentenceTransformer for encoding
    encoder_map = {
        "SentenceBert": "sentence-transformers/all-MiniLM-L6-v2",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-large": "intfloat/e5-large-v2",
        "roberta": "sentence-transformers/all-roberta-large-v1",
    }

    if args.encoder_name in encoder_map:
        model = SentenceTransformer(encoder_map[args.encoder_name], device=str(device))
        with torch.no_grad():
            text_embeddings = model.encode(
                graph_data.raw_texts,
                show_progress_bar=True,
                batch_size=64,
                convert_to_tensor=True,
            )
        generated_node_emb = text_embeddings.cpu()
    else:
        print(f"Encoder {args.encoder_name} requires custom TextEncoder implementation.")
        exit(1)

    print(f"[{args.dataset}-{args.encoder_name}] Node Embedding Shape {generated_node_emb.shape}")
    if args.save_emb:
        write_dir = f"../datasets/{args.encoder_name}"
        os.makedirs(write_dir, exist_ok=True)
        torch.save(generated_node_emb, f"{write_dir}/{args.dataset}.pt")

    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
