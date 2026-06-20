#!/bin/bash
# First-run smoke training for the atomic emoji vocabulary.
# Uses local emoji_reply JSONL as direct emoji -> emoji data and logs to wandb
# offline by default. After `wandb login`, run `wandb sync <run-dir>` to upload.
set -e

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb/cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb/config}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${MPLCONFIGDIR}"

EMOJI_CACHE_DIR=${EMOJI_CACHE_DIR:-/tmp/emoji_mdlm_data}
MAX_STEPS=${MAX_STEPS:-20}

python -u main.py \
  mode=train \
  data=emoji_reply \
  data.cache_dir="${EMOJI_CACHE_DIR}" \
  data.emoji_vocab_cache="${EMOJI_CACHE_DIR}/atomic_emoji_vocab.json" \
  model=tiny-emoji \
  model.length=32 \
  parameterization=subs \
  backbone=dit \
  strategy=single_device \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${MAX_STEPS}" \
  trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=1 \
  loader.global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=0 \
  loader.pin_memory=false \
  training.ema=0.999 \
  eval.compute_generative_perplexity=false \
  wandb.project=emoji-model \
  wandb.name=emoji_atomic_first_run \
  hydra.run.dir=outputs/emoji_atomic_first_run \
  checkpointing.resume_from_ckpt=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${MAX_STEPS}"
