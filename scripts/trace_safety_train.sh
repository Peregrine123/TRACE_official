#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_safety_train.txt"
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3-4B"
  # "Qwen/Qwen3-8B"
  # "meta-llama/Llama-3.1-8B"
  # "Qwen/Qwen3Guard-Gen-4B"
  # "Qwen/Qwen3-1.7B"
  # "Qwen/Qwen3-4B-Instruct-2507"
  "meta-llama/Llama-3.1-8B-Instruct"
)
BASE_MODEL_STATUS=False
sft=False

# Dataset configs
DATASET_NAME="agent_safety"
FILE_NAMES=(
  # "merged_for_tsne.jsonl"
  # "augment.jsonl"
  "Safiron.jsonl"
  # "rjg.jsonl"
  # "ASSE.jsonl"
  # "rjudge_clean.jsonl"
  # "rjg_GPT_align.jsonl"
  # "ASSE_GPT_align.jsonl"
)
# MemGen configs
TRAIN_METHOD="sft"
NGPU=2
# Augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8
LEARNING_RATE=2e-5
BATCH_SIZE=1
seeds=(
  42
  # 40
  # 100
)
PROMPT_LATENTS_LENS=(16)
for FILE_NAME in ${FILE_NAMES[@]}; do
  for seed in ${seeds[@]}; do
    for BASE_MODEL in ${MODELS[@]}; do
      for PROMPT_LATENTS_LEN in ${PROMPT_LATENTS_LENS[@]}; do
        REASONER_MODEL=${BASE_MODEL}
        WEAVER_MODEL=${BASE_MODEL}
        TRIGGER_MODEL=${BASE_MODEL}
        echo "============================================================"
        echo "[RUN] BASE_MODEL=${BASE_MODEL}"
        echo "[RUN] BASE_MODEL_STATUS=${BASE_MODEL_STATUS}"
        echo "[RUN] SFT=${sft}"
        echo "[RUN] DATASET=${FILE_NAME}"
        echo "============================================================"

        deepspeed \
          --include localhost:0,1 \
          --master_port=29600 \
          main_safety.py \
          --deepspeed configs/zero2_ngpus.yaml \
          --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
          --options \
          model.model_name ${REASONER_MODEL} \
          model.base_model_status ${BASE_MODEL_STATUS} \
          model.sft ${sft} \
          model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
          model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
          model.weaver.model_name ${WEAVER_MODEL} \
          model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
          model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
          model.trigger.model_name ${TRIGGER_MODEL} \
          model.trigger.active False \
          dataset.mode ${TRAIN_METHOD} \
          dataset.file ${FILE_NAME} \
          dataset.sft.val_ratio 0.3 \
          run.mode train_and_evaluate \
          run.train_weaver True \
          run.seed ${seed} \
          run.train_trigger False \
          run.train_weaver_method ${TRAIN_METHOD} \
          run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
          run.weaver.per_device_eval_batch_size ${BATCH_SIZE} \
          run.weaver.sft.num_train_epochs 3 \
          run.weaver.sft.learning_rate ${LEARNING_RATE} \
          run.weaver.sft.bf16 True \
          run.semantic_gate.embedder_type hf \
          run.semantic_gate.model_name ${REASONER_MODEL} \
          run.semantic_gate.enable_adjacent_turn_gate True \
          run.interaction.do_sample True \
          run.interaction.temperature 1.0 \
          run.interaction.max_response_length 4 \

      done
    done
  done
done

