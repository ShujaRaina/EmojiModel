#!/bin/bash
# Train Phase 1 emoji-only diffusion data on KomeijiForce/Text2Emoji.
# Text is not tokenized; sequences contain only atomic emoji plus controls.
#
# GPU (recommended): trains the `tiny-emoji` backbone end to end.
#   bash scripts/train_text2emoji.sh
#
# Scale up by swapping `model=small model.length=128` and raising
# `trainer.max_steps` / `loader.global_batch_size`.
set -e

python -u -m main \
  mode=train \
  data=text2emoji \
  model=tiny-emoji \
  model.length=64 \
  parameterization=subs \
  backbone=dit \
  wandb.name=emoji_run \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.max_steps=50000 \
  trainer.val_check_interval=2000 \
  eval.compute_generative_perplexity=False \
  sampling.steps=128
