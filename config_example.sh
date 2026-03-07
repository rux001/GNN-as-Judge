#!/bin/bash
# GNN-as-Judge Configuration Example
# Copy this file to config.sh and modify according to your setup

# ===== WORKSPACE CONFIGURATION =====
export WORKSPACE_DIR="/path/to/your/workspace"
export LF_DIR="/path/to/GNN_as_Judge/LLaMA-Factory"
export PROJECT_DIR="/path/to/GNN_as_Judge"
export ENV_NAME="GNNJudge"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

# ===== MODEL PATHS =====
export BASE_MODEL_PATH="/path/to/models/Meta-Llama-3-8B-Instruct"
export TEMPLATE="llama3"  # Options: llama3, mistral, llama2, qwen

# ===== GNN CONFIGURATION =====
export GNN_TYPE="GCN"           # Options: GCN, GAT, SAGE, SGConv
export GNN_HIDDEN_DIM=64
export GNN_LAYERS=2

# ===== TRAINING HYPERPARAMETERS =====
# SFT (Supervised Fine-Tuning)
export LEARNING_RATE_SFT=5e-6
export BATCH_SIZE_SFT=4
export EPOCHS_SFT=10

# DPO (Direct Preference Optimization)
export LEARNING_RATE_DPO=1e-5
export BATCH_SIZE_DPO=4
export EPOCHS_DPO=8
export DPO_BETA=0.1

# LoRA
export LORA_RANK=8
export LORA_ALPHA=16
export GRAD_ACCUM_STEPS=1

# ===== HARDWARE CONFIGURATION =====
export NUM_GPUS=4
export MAIN_PROCESS_PORT_BASE=29500
export VISIBLE_DEVICES="0,1,2,3"

# ===== NODE SELECTION PARAMETERS =====
export TOPK_INFLUENTIAL=1500
export CONFIDENCE_THRESHOLD=0.7
export MAX_SUBGRAPH_NODES=3000
