#!/bin/bash
# CPU-only training of the text -> emoji diffusion model. Useful for machines
# without a GPU (no flash-attn / mamba kernels required). This trains a small
# proof-of-concept model; use scripts/train_text2emoji.sh on a GPU for quality.
set -e

export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}

python -u main.py \
  mode=train \
  data=text2emoji \
  model=tiny-emoji \
  model.length=64 \
  parameterization=subs \
  backbone=dit \
  strategy=ddp \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  trainer.max_steps=${MAX_STEPS:-5000} \
  trainer.val_check_interval=1000 \
  trainer.limit_val_batches=10 \
  trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=50 \
  loader.global_batch_size=64 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  loader.num_workers=4 \
  training.ema=0.999 \
  eval.compute_generative_perplexity=False \
  wandb.name=emoji_run \
  hydra.run.dir=outputs/emoji_run \
  checkpointing.resume_from_ckpt=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000
