#!/bin/bash
# Two-phase atomic emoji run:
#   1. Use a Hugging Face text encoder only as offline scaffolding to build
#      semantic emoji->emoji pairs from Text2Emoji.
#   2. Resume from the Phase 1 checkpoint and tune on the repo emoji_reply
#      JSONL only, with optional in-loader permutation augmentation.
set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb/cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb/config}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export TOKENIZERS_PARALLELISM=false

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
  "${MPLCONFIGDIR}"

RUN_ROOT=${RUN_ROOT:-outputs/emoji_two_phase_h100}
DATA_CACHE=${DATA_CACHE:-/tmp/emoji_mdlm_two_phase}
TEACHER_MODEL=${TEACHER_MODEL:-BAAI/bge-small-en-v1.5}
SEMANTIC_LIMIT=${SEMANTIC_LIMIT:-5000}
SEMANTIC_TOP_K=${SEMANTIC_TOP_K:-3}
PHASE1_STEPS=${PHASE1_STEPS:-300}
PHASE2_STEPS=${PHASE2_STEPS:-500}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-512}
MODEL=${MODEL:-small}
MODEL_LENGTH=${MODEL_LENGTH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}

if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="$(pwd)/${RUN_ROOT}"
fi
if [[ "${DATA_CACHE}" != /* ]]; then
  DATA_CACHE="$(pwd)/${DATA_CACHE}"
fi

SEMANTIC_TABLE="${RUN_ROOT}/semantic_table.pt"
SEMANTIC_PAIRS="${RUN_ROOT}/semantic_pairs.jsonl"
VOCAB_CACHE="${DATA_CACHE}/atomic_emoji_vocab.json"
REPLY_DATA_FILE="$(pwd)/data/emoji_reply/emoji_reply.jsonl"
PHASE1_DIR="${RUN_ROOT}/phase1_semantic"
PHASE2_DIR="${RUN_ROOT}/phase2_reply"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE}" "${PHASE1_DIR}" "${PHASE2_DIR}"

python scripts/build_emoji_semantic_table.py \
  --cache-dir "${DATA_CACHE}" \
  --output "${SEMANTIC_TABLE}" \
  --pairs-output "${SEMANTIC_PAIRS}" \
  --teacher-model "${TEACHER_MODEL}" \
  --batch-size 512 \
  --top-k "${SEMANTIC_TOP_K}" \
  --limit "${SEMANTIC_LIMIT}" \
  --device cuda

COMMON_ARGS=(
  mode=train
  data=emoji_reply
  data.cache_dir="${DATA_CACHE}"
  data.emoji_vocab_cache="${VOCAB_CACHE}"
  'data.emoji_vocab_sources=[text2emoji,common]'
  "data.emoji_vocab_extra_files=[${REPLY_DATA_FILE}]"
  model="${MODEL}"
  model.length="${MODEL_LENGTH}"
  parameterization=subs
  backbone=dit
  trainer.accelerator=cuda
  trainer.devices=1
  trainer.precision=bf16-mixed
  +trainer.check_val_every_n_epoch=null
  trainer.limit_val_batches=2
  trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=5
  loader.global_batch_size="${GLOBAL_BATCH_SIZE}"
  loader.batch_size="${BATCH_SIZE}"
  loader.eval_batch_size="${BATCH_SIZE}"
  loader.num_workers="${NUM_WORKERS}"
  loader.pin_memory=true
  training.ema=0.9999
  eval.compute_generative_perplexity=false
  wandb.project=emoji-model
)

python -u -m main \
  "${COMMON_ARGS[@]}" \
  data.data_file="${SEMANTIC_PAIRS}" \
  data.emoji_include_challenge_in_train=false \
  data.permutation_augment_prob=0.0 \
  trainer.max_steps="${PHASE1_STEPS}" \
  trainer.val_check_interval="${PHASE1_VAL_INTERVAL:-50}" \
  wandb.name="emoji_phase1_semantic_h100_${RUN_SUFFIX}" \
  wandb.id="emoji_phase1_semantic_h100_${RUN_SUFFIX}" \
  hydra.run.dir="${PHASE1_DIR}" \
  checkpointing.resume_from_ckpt=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50

python -u -m main \
  "${COMMON_ARGS[@]}" \
  data.data_file="${REPLY_DATA_FILE}" \
  data.emoji_include_challenge_in_train=false \
  data.emoji_challenge_repeat=1 \
  data.permutation_augment_prob="${PERMUTATION_AUGMENT_PROB:-0.6}" \
  data.permutation_augment_prompt=true \
  data.permutation_augment_response=true \
  trainer.max_steps="${PHASE2_STEPS}" \
  trainer.val_check_interval="${PHASE2_VAL_INTERVAL:-50}" \
  wandb.name="emoji_phase2_reply_h100_${RUN_SUFFIX}" \
  wandb.id="emoji_phase2_reply_h100_${RUN_SUFFIX}" \
  hydra.run.dir="${PHASE2_DIR}" \
  checkpointing.resume_from_ckpt=false \
  +checkpointing.init_from_ckpt_path="${PHASE1_DIR}/checkpoints/last.ckpt" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50
