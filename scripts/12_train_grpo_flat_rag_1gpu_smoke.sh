#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

GRPO_CONFIG="${GRPO_CONFIG:-${REPO_DIR}/configs/grpo_flat_rag_1gpu_smoke.env}"

if [[ ! -f "${GRPO_CONFIG}" ]]; then
    die "缺少 GRPO 配置：${GRPO_CONFIG}"
fi

# shellcheck disable=SC1090
source "${GRPO_CONFIG}"

if [[ "${1:-}" == "--print-config" ]]; then
    env | grep '^GRPO_' | sort
    exit 0
fi

require_command conda
require_command curl

if [[ "${INDEX_PATH##*/}" != "e5_Flat.index" ]]; then
    die "GRPO RAG 仅允许使用 e5_Flat.index，当前为：${INDEX_PATH}"
fi

if [[ ! -s "${INDEX_PATH}" || ! -s "${CORPUS_PATH}" ]]; then
    die "缺少 Flat 索引或 Wiki-18 语料"
fi

if [[ ! -s "${RETRIEVER_MODEL_PATH}/config.json" ]]; then
    die "E5 检索模型目录无效：${RETRIEVER_MODEL_PATH}"
fi

if ! curl -fsS --max-time 10 "${RETRIEVER_URL%/retrieve}/openapi.json" >/dev/null; then
    die "检索服务未就绪：${RETRIEVER_URL}；请先运行 bash scripts/02_start_retriever.sh"
fi

if [[ ! -s "${QA_DATA_DIR}/train.parquet" || ! -s "${QA_DATA_DIR}/test.parquet" ]]; then
    die "缺少训练或验证 Parquet 数据集：${QA_DATA_DIR}"
fi

if [[ "${1:-}" == "--check" ]]; then
    printf '[PASS] Flat FAISS RAG GRPO preflight passed\n'
    printf 'index=%s\ncorpus=%s\nretriever=%s\n' \
        "${INDEX_PATH}" "${CORPUS_PATH}" "${RETRIEVER_URL}"
    exit 0
fi

run_tag="$(date +%Y%m%d_%H%M%S)"
run_label="${GRPO_RUN_LABEL:-grpo_flat_rag_1gpu_smoke}"
experiment_name="${run_label}_${run_tag}"
checkpoint_dir="${SEARCHRL_ROOT}/checkpoints/${experiment_name}"
log_file="${LOG_DIR}/${experiment_name}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED="${GRPO_PYTHONHASHSEED:-1000}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS="${GRPO_OMP_NUM_THREADS:-12}"
fi

log "GRPO 配置：${GRPO_CONFIG}"
log "检索：${RETRIEVER_MODEL_PATH} + ${INDEX_PATH}"
log "训练数据：${QA_DATA_DIR}/train.parquet（前 ${GRPO_TRAIN_DATA_NUM} 条）"
log "输出：${checkpoint_dir}"
log "日志：${log_file}"

cd "${SEARCHR1_CODE_DIR}"

conda run --no-capture-output -n searchr1 \
    python -m verl.trainer.main_ppo_format \
    data.train_files="${QA_DATA_DIR}/train.parquet" \
    data.val_files="${QA_DATA_DIR}/test.parquet" \
    data.train_data_num="${GRPO_TRAIN_DATA_NUM}" \
    data.val_data_num="${GRPO_VAL_DATA_NUM}" \
    data.train_batch_size="${GRPO_TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${GRPO_VAL_BATCH_SIZE}" \
    data.max_prompt_length="${GRPO_MAX_PROMPT_LENGTH}" \
    data.max_response_length="${GRPO_MAX_RESPONSE_LENGTH}" \
    data.max_start_length="${GRPO_MAX_START_LENGTH}" \
    data.max_obs_length="${GRPO_MAX_OBS_LENGTH}" \
    data.shuffle_train_dataloader="${GRPO_SHUFFLE_TRAIN_DATALOADER:-true}" \
    algorithm.adv_estimator=grpo \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="${GRPO_LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload="${GRPO_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.grad_offload="${GRPO_GRAD_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${GRPO_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload="${GRPO_REF_PARAM_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GRPO_VLLM_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n_agent="${GRPO_N_AGENTS}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    trainer.logger='[wandb]' \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="${GRPO_SAVE_FREQ:--1}" \
    +trainer.save_final_checkpoint="${GRPO_SAVE_FINAL_CHECKPOINT:-true}" \
    trainer.test_freq="${GRPO_TEST_FREQ:--1}" \
    trainer.project_name=searchrl-eval-grpo \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${GRPO_TOTAL_STEPS}" \
    trainer.default_local_dir="${checkpoint_dir}" \
    reward_model.structure_format_score="${GRPO_STRUCTURE_FORMAT_SCORE}" \
    reward_model.final_format_score="${GRPO_FINAL_FORMAT_SCORE}" \
    reward_model.retrieval_score="${GRPO_RETRIEVAL_SCORE}" \
    max_turns="${GRPO_MAX_TURNS}" \
    do_search=true \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk="${RETRIEVER_TOPK}" \
    2>&1 | tee "${log_file}"
