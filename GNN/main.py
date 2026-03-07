import argparse
import time
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import (
    load_graph_dataset,
    GNNEncoder,
    array_mean_std,
    compute_acc_and_f1,
    set_seed,
    create_few_shot_dataset,
    load_graph_dataset_re_split,
    load_graph_dataset_for_gnn,
    HeteroGNNEncoder,
    plain_adj_matrix,
)
import torch.nn.functional as F


DEFAULT_LM, DEFAULT_LLM = "roberta", "Mistral-7B"


def gnn_train():
    gnn_model.train()
    optimizer.zero_grad()
    output = gnn_model(graph_data.x, graph_data.edge_index)
    loss = F.cross_entropy(output[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def gnn_train_fullbatch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def gnn_test():
    gnn_model.eval()
    pred = gnn_model(graph_data.x, graph_data.edge_index).argmax(dim=1)

    accuracy, macro_f1_scores, micro_f1_scores = [], [], []
    for mask in [graph_data.train_mask, graph_data.val_mask, graph_data.test_mask]:
        acc, macro_f1, micro_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), graph_data.y[mask].cpu().numpy())
        accuracy.append(acc)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)
        
    return accuracy, macro_f1_scores, micro_f1_scores


@torch.no_grad()
def gnn_test_fullbatch(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)

    accuracy, macro_f1_scores, micro_f1_scores = [], [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        if mask.sum() == 0:
            acc, macro_f1, micro_f1 = 0.0, 0.0, 0.0
        else:
            acc, macro_f1, micro_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), data.y[mask].cpu().numpy())
        accuracy.append(acc)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)
        
    return accuracy, macro_f1_scores, micro_f1_scores


def print_dataset_stats(graph_data):
    total_nodes = graph_data.x.shape[0]
    num_train = int(torch.sum(graph_data.train_mask))
    num_val = int(torch.sum(graph_data.val_mask))
    num_test = int(torch.sum(graph_data.test_mask))
    
    print(f"Data shape: features {graph_data.x.shape}, edges {graph_data.edge_index.shape}")
    print(f"Total nodes: {total_nodes}")
    print(f"Train nodes: {num_train} ({num_train/total_nodes:.2%})")
    print(f"Validation nodes: {num_val} ({num_val/total_nodes:.2%})")
    print(f"Test nodes: {num_test} ({num_test/total_nodes:.2%})")

    if hasattr(graph_data, "y") and hasattr(graph_data, "label_name"):
        num_classes = len(graph_data.label_name)
        train_labels = graph_data.y[graph_data.train_mask]
        val_labels = graph_data.y[graph_data.val_mask]
        test_labels = graph_data.y[graph_data.test_mask]
        train_label_counts = torch.bincount(train_labels, minlength=num_classes)
        val_label_counts = torch.bincount(val_labels, minlength=num_classes)
        test_label_counts = torch.bincount(test_labels, minlength=num_classes)
        print(f"Train label distribution: {train_label_counts.tolist()}")
        print(f"Validation label distribution: {val_label_counts.tolist()}")
        print(f"Test label distribution: {test_label_counts.tolist()}")
    else:
        num_classes = int(graph_data.y.max().item() + 1)
    
    return num_classes


def train_heterognn(gnn_model, graph_data, optimizer, args, model_save_path, run_idx):
    print("Using HeteroGNN with full-batch training.")
    
    best_eval_acc = best_test_acc = 0.0
    best_eval_mac_f1 = best_test_mac_f1 = 0.0
    best_eval_weight_f1 = best_test_weight_f1 = 0.0
    st_time, counter = time.time(), 0

    for epoch in range(1, args.epochs + 1):
        cur_loss = gnn_train_fullbatch(gnn_model, graph_data, optimizer)
        [train_acc, val_acc, test_acc], [train_mac_f1, val_mac_f1, test_mac_f1], [_, _, test_weight_f1] = gnn_test_fullbatch(gnn_model, graph_data)

        if val_acc > best_eval_acc:
            best_eval_acc, best_test_acc = val_acc, test_acc
            best_eval_mac_f1, best_test_mac_f1 = val_mac_f1, test_mac_f1
            best_eval_weight_f1, best_test_weight_f1 = val_mac_f1, test_weight_f1
            counter = 0
            torch.save(gnn_model.state_dict(), model_save_path)
            print(f"Epoch {epoch:03d}: New best validation accuracy {val_acc:.3f}. Model saved to {model_save_path}.")
        else:
            counter += 1

        if epoch % args.print_freq == 0:
            print(f"Epoch {epoch:03d}   Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_mac_f1:.3f} Val F1 {val_mac_f1:.3f} Test F1 {test_mac_f1:.3f}")
        
        if counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return best_test_acc, best_test_mac_f1, best_test_weight_f1, round(time.time() - st_time, 3)


def train_standard_gnn(gnn_model, graph_data, optimizer, args, model_save_path, run_idx):
    best_eval_acc = best_test_acc = 0.0
    best_eval_mac_f1 = best_test_mac_f1 = 0.0
    best_eval_weight_f1 = best_test_weight_f1 = 0.0
    st_time, counter = time.time(), 0

    for epoch in range(1, args.epochs+1):
        cur_loss = gnn_train()
        [train_acc, val_acc, test_acc], [train_mac_f1, val_mac_f1, test_mac_f1], [train_weight_f1, val_weight_f1, test_weight_f1] = gnn_test()
        
        if val_acc > best_eval_acc:
            best_eval_acc, best_test_acc = val_acc, test_acc
            best_eval_mac_f1, best_test_mac_f1 = val_mac_f1, test_mac_f1
            best_eval_weight_f1, best_test_weight_f1 = val_weight_f1, test_weight_f1
            counter = 0
            torch.save(gnn_model.state_dict(), model_save_path)
            print(f"Epoch {epoch:03d}: New best validation accuracy {val_acc:.3f}. Model saved to {model_save_path}.")
        else:
            counter += 1
        
        if epoch % args.print_freq == 0:
            print(f"Epoch {epoch:03d}   Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_mac_f1:.3f} Val F1 {val_mac_f1:.3f} Test F1 {test_mac_f1:.3f}")
        
        if counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return best_test_acc, best_test_mac_f1, best_test_weight_f1, round(time.time() - st_time, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--encoder_name", type=str, default="", choices=["", "shallow", "LM", "LLM", "e5-large", "SentenceBert", "MiniLM", "roberta", "Qwen-3B", "Mistral-7B", "Qwen-7B", "Llama-8B"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--gnn_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE", "TransformerConv", "SGConv", "HeteroGNN"])
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_softmax", type=int, default=0)
    parser.add_argument("--residual_conn", type=int, default=0)
    parser.add_argument("--jump_knowledge", type=int, default=0)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--label_rate", type=float, default=0.05)
    parser.add_argument("--re_split", type=int, default=0)
    parser.add_argument("--write_result", type=int, default=1)
    
    args = parser.parse_args()
    print(args)
    
    device = torch.device(args.device)
    if args.encoder_name == "LM":
        args.encoder_name = DEFAULT_LM 
    elif args.encoder_name == "LLM":
        args.encoder_name = DEFAULT_LLM

    final_acc_list, final_macro_f1_list, final_weight_f1_list, timer_list = [], [], [], []

    os.makedirs("../results/GNN", exist_ok=True)
    
    if args.write_result:
        write_file = open(f"../results/GNN/{args.dataset}_{args.shots}_shot{'' if not args.re_split else '_s'}.csv", 
                         mode='a', newline='')
        
    for i in range(args.run_times):
        current_seed = args.seed + i
        set_seed(current_seed)
        print(f"\n=== Running with seed {current_seed} (Run {i+1}/{args.run_times}) ===\n")

        graph_data = create_few_shot_dataset(
            args.dataset,
            shots=args.shots,
            seed=current_seed,
            device=device,
            path_prefix=".."
        )
        graph_data = graph_data.to(device)

        num_classes = print_dataset_stats(graph_data)

        if args.gnn_type == "HeteroGNN":
            model_save_path = f"../results/GNN/{args.dataset}_{args.gnn_type}_{args.shots}_shot_best_model_run{i}.pt"
        else:
            model_save_path = f"../results/GNN/{args.dataset}_{args.shots}_shot_best_model_run{i}.pt"
        
        if args.gnn_type == "HeteroGNN":
            graph_data.edge_index = plain_adj_matrix(graph_data.edge_index, graph_data.num_nodes).to(device)
            gnn_model = HeteroGNNEncoder(
                input_dim=graph_data.x.shape[1],
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                n_layers=args.n_layers, 
                dropout=args.dropout
            ).to(device)
        else:
            gnn_model = GNNEncoder(
                input_dim=graph_data.x.shape[1],
                hidden_dim=args.hidden_dim, 
                output_dim=num_classes,
                n_layers=args.n_layers,
                gnn_type=args.gnn_type,
                dropout=args.dropout,
                use_softmax=args.use_softmax,
                batch_norm=args.batch_norm,
                residual_conn=args.residual_conn,
                jump_knowledge=args.jump_knowledge
            ).to(device)
        
        if i == 0:
            trainable_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
            print(f"[GNN] Number of parameters: {trainable_params:,}")

        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        if args.gnn_type == "HeteroGNN":
            best_test_acc, best_test_mac_f1, best_test_weight_f1, training_time = train_heterognn(
                gnn_model, graph_data, optimizer, args, model_save_path, i
            )
        else:
            best_test_acc, best_test_mac_f1, best_test_weight_f1, training_time = train_standard_gnn(
                gnn_model, graph_data, optimizer, args, model_save_path, i
            )
        
        timer_list.append(training_time)
        print(f'[Times {i}] Test Acc {best_test_acc:.2f}  Test Macro-F1 {best_test_mac_f1:.2f}  Test Micro-F1 {best_test_weight_f1:.2f} Time {training_time:.3f}s\n')
        final_acc_list.append(best_test_acc)
        final_macro_f1_list.append(best_test_mac_f1)
        final_weight_f1_list.append(best_test_weight_f1)
    
    acc_mean, acc_std = array_mean_std(final_acc_list)
    macrof1_mean, macrof1_std = array_mean_std(final_macro_f1_list)
    weightf1_mean, weightf1_std = array_mean_std(final_weight_f1_list)
    print(f"\n[Final] Acc {acc_mean}±{acc_std}  Macro-F1 {macrof1_mean}±{macrof1_std}  Weight-F1 {weightf1_mean}±{weightf1_std}")

    if args.write_result:
        import csv
        writer = csv.writer(write_file)
        writer.writerow([
            args.gnn_type, args.n_layers, args.hidden_dim, args.dropout, 
            args.encoder_name, args.batch_norm, args.residual_conn, 
            f"{acc_mean:.2f}±{acc_std:.2f}", f"{macrof1_mean:.2f}±{macrof1_std:.2f}", 
            f"{weightf1_mean:.2f}±{weightf1_std:.2f}", trainable_params, 
            f"{sum(timer_list)/len(timer_list):.3f}s"
        ])
        write_file.close()