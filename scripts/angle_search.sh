#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_safety_eval.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
BASE_MODELS=(
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-4B-Instruct-2507"
    # "Qwen/Qwen3-4B-Instruct-2507"
    # "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-8B"
    # "Qwen/Qwen3Guard-Gen-4B"
    # "Qwen/Qwen3Guard-Gen-4B"
    # "Qwen/Qwen3Guard-Gen-4B"
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
)
LOAD_MODEL_PATHS=(
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-1.5B-Instruct/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260109-131351/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-1.5B-Instruct/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-005143/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-1.5B-Instruct/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260108-230223/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-3B-Instruct/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260109-133108/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-3B-Instruct/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-014057/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-3B-Instruct/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260109-004645/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-7B-Instruct/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260109-135541/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-7B-Instruct/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-024418/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen2.5-7B-Instruct/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260109-022303/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-1.7B/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260112-132008/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-1.7B/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260112-133522/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-1.7B/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260112-131109/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-4B/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260109-142223/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-4B/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-035129/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-4B/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260109-040556/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-4B-Instruct-2507/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260112-145339/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-4B-Instruct-2507/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260112-151511/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-4B-Instruct-2507/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260112-144039/weaver
PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-8B/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260110-151427/weaver
PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-8B/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-050748/weaver
PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3-8B/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260109-060248/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3Guard-Gen-4B/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260111-232020/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3Guard-Gen-4B/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-235354/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Qwen3Guard-Gen-4B/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260112-171040/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Llama-3.1-8B/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260109-145903/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Llama-3.1-8B/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260111-062923/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Llama-3.1-8B/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260109-072515/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Llama-3.1-8B-Instruct/ASSE/ASSE_bms_True_pn=1_pl=16_in=5_il=8_20260112-172524/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Llama-3.1-8B-Instruct/ASSE_GPT_align/ASSE_GPT_align_bms_True_pn=1_pl=16_in=5_il=8_20260112-175134/weaver
# PATH_TO_CHECKPOINT/results/train/agent_safety/Llama-3.1-8B-Instruct/rjg/rjg_bms_True_pn=1_pl=16_in=5_il=8_20260112-164520/weaver
)

DATASETS=(
"ASSE.jsonl"
"ASSE_GPT_align.jsonl"
"rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
# "ASSE.jsonl"
# "ASSE_GPT_align.jsonl"
# "rjg.jsonl"
)
NUM=${#BASE_MODELS[@]}
BASE_MODEL_STATUS=True    ################## 记得改 ######################
sft=False
angles=(
    0
    5
    10
    15
    20
    25
    30
    35
    40
    45
    50
    55
    60
    65
    70
    75
    80
    85
    90
)
# Dataset configs
DATASET_NAME="agent_safety"
# FILE_NAME="ASSE.jsonl"
# Augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

BATCH_SIZE=8
# Trained model path

# evaluate
for angle in ${angles[@]}; do
    for ((i=0; i<NUM; i++)); do
        BASE_MODEL="${BASE_MODELS[$i]}"
        LOAD_MODEL_PATH="${LOAD_MODEL_PATHS[$i]}"
        FILE_NAME="${DATASETS[$i]}"
        REASONER_MODEL=${BASE_MODEL}
        WEAVER_MODEL=${BASE_MODEL}
        TRIGGER_MODEL=${BASE_MODEL}
        echo "  Training module $i"
        echo "  BASE_MODEL=$BASE_MODEL"
        echo "  LOAD_MODEL_PATH=$LOAD_MODEL_PATH"
        deepspeed \
            --include localhost:2\
            --master_port=29603 \
            main_safety.py \
            --deepspeed configs/zero2_ngpus.yaml \
            --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
            --options \
            model.model_name ${REASONER_MODEL} \
            model.base_model_status ${BASE_MODEL_STATUS} \
            model.sft ${sft} \
            model.load_model_path ${LOAD_MODEL_PATH} \
            model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
            model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
            model.weaver.model_name ${WEAVER_MODEL} \
            model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
            model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
            model.trigger.model_name ${TRIGGER_MODEL} \
            model.trigger.active False \
            dataset.file ${FILE_NAME} \
            dataset.sft.val_ratio 0.3 \
            run.mode evaluate \
            run.semantic_gate.embedder_type hf \
            run.semantic_gate.max_length 1024 \
            run.semantic_gate.model_name ${REASONER_MODEL} \
            run.semantic_gate.angle_threshold_deg ${angle} \
            run.semantic_gate.entropy_threshold 10 \
            run.semantic_gate.enable_adjacent_turn_gate True \
            run.interaction.batch_size ${BATCH_SIZE} \
            run.interaction.do_sample True \
            run.interaction.temperature 1.0 \
            run.interaction.max_response_length 4 \
            run.interaction.angle_search True \

  done
done