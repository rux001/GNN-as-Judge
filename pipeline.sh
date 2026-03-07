# ===== CONFIGURATION =====
# Source your config file or set these variables directly
# See config_example.sh for all available options
if [ -f "config.sh" ]; then
    source config.sh
fi

# Command line arguments or defaults
DATASET=${1:-"cora"}
SHOT_COUNT=${2:-"5"}
SEED=${3:-"42"}


# ===== ENVIRONMENT SETUP =====
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES:-"0"}
export HF_HOME="$WORKSPACE_DIR/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Create directories
MAIN_EXP_DIR="$WORKSPACE_DIR/results/gnn_as_judge"
DATASET_DIR="$LF_DIR/data"
DATASET_INFO_FILE="$DATASET_DIR/dataset_info.json"
mkdir -p "$HF_HOME" "$MAIN_EXP_DIR" "$DATASET_DIR"

# Activate environment
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${ENV_NAME:-GNNJudge}"

# ===== PIPELINE EXECUTION =====
RUN_ID="${DATASET}_${SHOT_COUNT}shot_seed${SEED}"
RUN_DIR="$MAIN_EXP_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

# Directory structure
SFT_LOG_DIR="$RUN_DIR/sft/logs"
SFT_OUTDIR="$RUN_DIR/sft/model"
SFT_RESULTS_DIR="$RUN_DIR/sft/predictions"
DPO_LOG_DIR="$RUN_DIR/dpo/logs"
DPO_OUTDIR="$RUN_DIR/dpo/model"
DPO_RESULTS_DIR="$RUN_DIR/dpo/predictions"
mkdir -p "$SFT_LOG_DIR" "$SFT_OUTDIR" "$SFT_RESULTS_DIR" "$DPO_LOG_DIR" "$DPO_OUTDIR" "$DPO_RESULTS_DIR"

# File paths
SFT_DATASET_PREFIX="${DATASET}_sft_${SHOT_COUNT}_shot"
LLM_PRED_FILE="$RUN_DIR/${RUN_ID}_llm_preds.jsonl"
SELECTED_NODES_FILE="$RUN_DIR/${RUN_ID}_selected_nodes.json"
DPO_DATASET_NAME="${RUN_ID}_dpo"
DPO_JSON_FILE="$DATASET_DIR/${DPO_DATASET_NAME}.json"
SFT_DPO_JSON_FILE="$DATASET_DIR/${DPO_DATASET_NAME}_sft.json"
GNN_MODEL_PATH="$PROJECT_DIR/results/GNN/${DATASET}_${SHOT_COUNT}_shot_best_model_run0.pt"

echo "=== Starting GNN-as-Judge Pipeline for $RUN_ID ==="

# STAGE 1: Create SFT Dataset
echo "--- Stage 1: Create SFT Dataset ---"
cd "$PROJECT_DIR"
python create_sft.py \
  --dataset "$DATASET" \
  --output "$DATASET_DIR/${DATASET}_sft.json" \
  --shots "$SHOT_COUNT" \
  --seed "$SEED" \
  --path_prefix "."

# Update dataset_info.json for SFT
python - <<EOF
import json, os
p = "$DATASET_INFO_FILE"
prefix = "$SFT_DATASET_PREFIX"
os.makedirs(os.path.dirname(p), exist_ok=True)
info = json.load(open(p)) if os.path.exists(p) else {}
for split in ["train", "val", "test", "unlabeled"]:
    key = f"{prefix}_{split}"
    info[key] = {
        "file_name": f"{key}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"}
    }
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
print("Dataset info updated for SFT")
EOF

# STAGE 2: Train SFT Model
echo "--- Stage 2: Train SFT Model ---"
cd "$LF_DIR"
CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES python -m accelerate.commands.accelerate_cli launch \
  --config_file "$LF_DIR/examples/accelerate/single_config.yaml" \
  --num_processes ${NUM_GPUS:-1} \
  --main_process_port ${MAIN_PROCESS_PORT_BASE:-29500} \
  "$LF_DIR/src/train.py" \
  --stage sft --do_train \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --dataset "${SFT_DATASET_PREFIX}_train" \
  --template ${TEMPLATE:-"llama3"} \
  --finetuning_type lora \
  --lora_rank ${LORA_RANK:-8} --lora_alpha ${LORA_ALPHA:-16} --lora_target all \
  --output_dir "$SFT_OUTDIR" --overwrite_cache --overwrite_output_dir \
  --cutoff_len 2048 --preprocessing_num_workers 16 \
  --per_device_train_batch_size ${BATCH_SIZE_SFT:-4} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS:-2} \
  --lr_scheduler_type cosine --logging_steps 20 --save_steps 500 \
  --learning_rate ${LEARNING_RATE_SFT:-5e-5} --num_train_epochs ${EPOCHS_SFT:-3} \
  --plot_loss --bf16 --save_total_limit 3 \
  --logging_dir "$SFT_LOG_DIR" 2>&1 | tee "$SFT_LOG_DIR/train.log"

# Find best SFT checkpoint (by modification time, not name, to avoid stale checkpoints)
BEST_SFT_CHECKPOINT="$SFT_OUTDIR"
if [ -d "$SFT_OUTDIR" ]; then
  LATEST_CHECKPOINT=$(ls -dt "$SFT_OUTDIR"/checkpoint-* 2>/dev/null | head -n 1)
  if [ -n "$LATEST_CHECKPOINT" ]; then
    BEST_SFT_CHECKPOINT="$LATEST_CHECKPOINT"
  fi
fi
echo "Using SFT checkpoint: $BEST_SFT_CHECKPOINT"

# STAGE 3: Select Influential Nodes
echo "--- Stage 3: Select Influential Nodes ---"
cd "$PROJECT_DIR"
python select_influential_nodes.py \
  --dataset "$DATASET" \
  --k ${TOPK_INFLUENTIAL:-100} \
  --output_file "$SELECTED_NODES_FILE" \
  --shots "$SHOT_COUNT" \
  --seed "$SEED" \
  --method auto \
  --max_subgraph_nodes ${MAX_SUBGRAPH_NODES:-3000} \
  --max_distance 3 \
  --path_prefix "."

# STAGE 3.5: Filter unlabeled dataset to selected nodes only
echo "--- Stage 3.5: Filter to Selected Nodes ---"
SELECTED_DATASET_NAME="${SFT_DATASET_PREFIX}_selected"
SELECTED_DATASET_FILE="$DATASET_DIR/${SELECTED_DATASET_NAME}.json"
cd "$PROJECT_DIR"
python - <<EOF
import json, os
selected_path = "$SELECTED_NODES_FILE"
node_ids_path = "$DATASET_DIR/${DATASET}_${SHOT_COUNT}_shot_unlabeled_node_ids.json"
unlabeled_path = "$DATASET_DIR/${SFT_DATASET_PREFIX}_unlabeled.json"
output_path = "$SELECTED_DATASET_FILE"
info_path = "$DATASET_INFO_FILE"

with open(selected_path) as f:
    selected_ids = set(json.load(f)['selected_node_ids'])
with open(node_ids_path) as f:
    all_ids = json.load(f)['selected_node_ids']
with open(unlabeled_path) as f:
    unlabeled = json.load(f)

filtered = []
ordered_ids = []
for i, nid in enumerate(all_ids):
    if nid in selected_ids:
        filtered.append(unlabeled[i])
        ordered_ids.append(nid)
with open(output_path, 'w') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)
# Save ordered node IDs (matching prediction file order) for create_wsft
ordered_path = "$SELECTED_NODES_FILE".replace('.json', '_ordered.json')
with open(ordered_path, 'w') as f:
    json.dump({"selected_node_ids": ordered_ids}, f, indent=2)
print(f"Filtered {len(filtered)} samples from {len(unlabeled)} unlabeled nodes")
print(f"Saved ordered node IDs to {ordered_path}")

info = json.load(open(info_path))
info["$SELECTED_DATASET_NAME"] = {
    "file_name": os.path.basename(output_path),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
json.dump(info, open(info_path, 'w'), indent=2, ensure_ascii=False)
print("Dataset info updated for selected nodes")
EOF

# STAGE 4: Generate LLM Predictions on Selected Nodes
echo "--- Stage 4: Generate LLM Predictions ---"
cd "$LF_DIR"
CUDA_VISIBLE_DEVICES=0 python src/vllm_infer.py \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --adapter_name_or_path "$BEST_SFT_CHECKPOINT" \
  --dataset "$SELECTED_DATASET_NAME" \
  --template ${TEMPLATE:-"llama3"} \
  --dataset_dir "$DATASET_DIR" \
  --save_name "$LLM_PRED_FILE"

# STAGE 5: Create DPO Dataset using GNN-as-Judge
echo "--- Stage 5: Create DPO Dataset ---"
cd "$PROJECT_DIR"
python create_wsft.py \
  --dataset "$DATASET" \
  --selected_nodes_path "${SELECTED_NODES_FILE%.json}_ordered.json" \
  --pretrained_model "$GNN_MODEL_PATH" \
  --llm_predictions "$LLM_PRED_FILE" \
  --dpo_output_path "$DPO_JSON_FILE" \
  --sft_output_path "$SFT_DPO_JSON_FILE" \
  --confidence_threshold ${CONFIDENCE_THRESHOLD:-0.7} \
  --shots "$SHOT_COUNT" \
  --gnn_type "${GNN_TYPE:-GCN}" \
  --hidden_dim ${GNN_HIDDEN_DIM:-64} \
  --n_layers ${GNN_LAYERS:-2} \
  --seed "$SEED" \
  --device "cuda:0"

# Update dataset_info.json for DPO
python - <<EOF
import json, os
p = "$DATASET_INFO_FILE"
ds = "$DPO_DATASET_NAME"
sft_ds = ds + "_sft"
info = json.load(open(p)) if os.path.exists(p) else {}
info[ds] = {
    "file_name": ds + ".json",
    "formatting": "sharegpt",
    "ranking": True,
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}
info[sft_ds] = {
    "file_name": sft_ds + ".json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
print("Dataset info updated for DPO")
EOF

# STAGE 6: DPO Training
echo "--- Stage 6: DPO Training ---"
cd "$LF_DIR"
CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES python -m accelerate.commands.accelerate_cli launch \
  --config_file "$LF_DIR/examples/accelerate/single_config.yaml" \
  --num_processes ${NUM_GPUS:-1} \
  --main_process_port $((${MAIN_PROCESS_PORT_BASE:-29500} + 1)) \
  "$LF_DIR/src/train.py" \
  --stage dpo --do_train \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --adapter_name_or_path "$BEST_SFT_CHECKPOINT" \
  --create_new_adapter \
  --dataset_dir "$DATASET_DIR" \
  --dataset "$DPO_DATASET_NAME" \
  --template ${TEMPLATE:-"llama3"} \
  --finetuning_type lora \
  --lora_rank ${LORA_RANK:-8} --lora_alpha ${LORA_ALPHA:-16} --lora_target all \
  --pref_beta ${DPO_BETA:-0.1} --pref_loss orpo \
  --output_dir "$DPO_OUTDIR" --overwrite_cache --overwrite_output_dir \
  --cutoff_len 2048 --preprocessing_num_workers 16 \
  --per_device_train_batch_size ${BATCH_SIZE_DPO:-2} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS:-2} \
  --lr_scheduler_type cosine --logging_steps 20 --save_steps 100 \
  --learning_rate ${LEARNING_RATE_DPO:-1e-6} --num_train_epochs ${EPOCHS_DPO:-1} \
  --plot_loss --bf16 --save_total_limit 3 \
  --logging_dir "$DPO_LOG_DIR" 2>&1 | tee "$DPO_LOG_DIR/train.log"

BEST_DPO_CHECKPOINT="$DPO_OUTDIR"
if [ -d "$DPO_OUTDIR" ]; then
  LATEST_CHECKPOINT=$(ls -dt "$DPO_OUTDIR"/checkpoint-* 2>/dev/null | head -n 1)
  if [ -n "$LATEST_CHECKPOINT" ]; then
    BEST_DPO_CHECKPOINT="$LATEST_CHECKPOINT"
  fi
fi

# STAGE 7: Final Evaluation
echo "--- Stage 7: Final Evaluation ---"
cd "$LF_DIR"
DPO_TEST_PRED_FILE="$DPO_RESULTS_DIR/dpo_test_predictions.jsonl"
CUDA_VISIBLE_DEVICES=0 python src/vllm_infer.py \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --adapter_name_or_path "$BEST_DPO_CHECKPOINT" \
  --dataset "${SFT_DATASET_PREFIX}_test" \
  --template ${TEMPLATE:-"llama3"} \
  --dataset_dir "$DATASET_DIR" \
  --save_name "$DPO_TEST_PRED_FILE"

cd "$PROJECT_DIR"
python evaluate_predictions.py \
  --dataset "$DATASET" \
  --pred_file "$DPO_TEST_PRED_FILE" \
  --output_dir "$DPO_RESULTS_DIR/final_eval" \
  --model_name "best_dpo" \
  --path_prefix "."

echo "=== GNN-as-Judge Pipeline Completed ==="
echo "Results saved to: $DPO_RESULTS_DIR/final_eval"
echo "Run ID: $RUN_ID"

# Extract and display final metrics
FINAL_METRICS_FILE="$DPO_RESULTS_DIR/final_eval/best_dpo/metrics.json"
if [ -f "$FINAL_METRICS_FILE" ]; then
  echo "Final Results:"
  python -c "
import json
try:
    with open('$FINAL_METRICS_FILE', 'r') as f:
        metrics = json.load(f)
    print(f\"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\")
    print(f\"  Macro-F1: {metrics.get('macro_f1', 'N/A'):.4f}\")
    print(f\"  Total Samples: {metrics.get('total_samples', 'N/A')}\")
except Exception as e:
    print(f'Error reading metrics: {e}')
"
fi
