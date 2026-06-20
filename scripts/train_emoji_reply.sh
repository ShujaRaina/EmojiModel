#!/bin/bash
# Train Phase 2 emoji-to-emoji generation.
#
# To use your own instruction-tuning set:
#   EMOJI_REPLY_DATA_FILE=/path/to/emoji_instructions.jsonl \
#     bash scripts/train_emoji_reply.sh
#
# Supported columns include prompt_emoji/response_emoji, source/target,
# input/output, prompt/response, or instruction/output. Non-emoji text is
# filtered out by the atomic tokenizer.
set -e

DATA_FILE_OVERRIDE=()
if [ -n "${EMOJI_REPLY_DATA_FILE}" ]; then
  DATA_FILE_OVERRIDE=(data.data_file="${EMOJI_REPLY_DATA_FILE}")
fi

python -u -m main \
  mode=train \
  data=emoji_reply \
  "${DATA_FILE_OVERRIDE[@]}" \
  model=tiny-emoji \
  model.length=64 \
  parameterization=subs \
  backbone=dit \
  wandb.name=emoji_reply_run \
  loader.global_batch_size=256 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.max_steps=50000 \
  trainer.val_check_interval=2000 \
  eval.compute_generative_perplexity=False \
  sampling.steps=128
