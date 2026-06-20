# Emoji MDLM hard1 results

## Run

- Date: 2026-06-20
- Model: `medium`, 328M params, length 64
- Vocab: atomic emoji tokens, shared across both phases
- Phase 1: semantic scaffolding from Text2Emoji/BGE pairs
- Phase 2: repo reply data only, `data/emoji_reply/emoji_reply.jsonl`
- Permutation augmentation: `PERMUTATION_AUGMENT_PROB=1.0`
- Batch: global 512, per-device 512
- Checkpoints:
  - `outputs/emoji_two_phase_h100_hard1/phase2_reply/checkpoints/best.ckpt`
  - `outputs/emoji_two_phase_h100_hard1/phase2_reply/checkpoints/last.ckpt`

Command:

```bash
RUN_SUFFIX=hard1 RUN_ROOT=outputs/emoji_two_phase_h100_hard1 DATA_CACHE=/tmp/emoji_mdlm_two_phase_hard1 MODEL=medium MODEL_LENGTH=64 SEMANTIC_LIMIT=20000 SEMANTIC_TOP_K=5 PHASE1_STEPS=1000 PHASE2_STEPS=3000 PHASE1_VAL_INTERVAL=100 PHASE2_VAL_INTERVAL=100 PERMUTATION_AUGMENT_PROB=1.0 GLOBAL_BATCH_SIZE=512 BATCH_SIZE=512 NUM_WORKERS=8 bash scripts/run_two_phase_h100.sh
```

WandB:

- Phase 1: https://wandb.ai/derektang-the-university-of-chicago/emoji-model/runs/emoji_phase1_semantic_h100_hard1
- Phase 2: https://wandb.ai/derektang-the-university-of-chicago/emoji-model/runs/emoji_phase2_reply_h100_hard1

## Training Signal

Phase 1 semantic validation NLL improved from `8.15573` at step 100 to `3.70475` at step 1000.

Phase 2 reply validation NLL reached best `2.92683` at step 200, then later checkpoints did not beat that validation loss. The final checkpoint is still useful for the thesis eval because it showed better permutation stability.

## Repo-Derived Permutation Eval

Eval command:

```bash
python scripts/eval_permutation_stability.py \
  --checkpoint hard1_best=outputs/emoji_two_phase_h100_hard1/phase2_reply/checkpoints/best.ckpt \
  --checkpoint hard1_last=outputs/emoji_two_phase_h100_hard1/phase2_reply/checkpoints/last.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_hard1 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json \
  --source reply_data \
  --reply-data-file data/emoji_reply/emoji_reply.jsonl \
  --num-groups 24 \
  --permutations-per-group 3 \
  --steps 32 \
  --device cuda \
  --model medium \
  --jsonl-out outputs/emoji_two_phase_h100_hard1/eval/reply_permutation_best_last_results.jsonl
```

Metrics:

| model | perm_stability | target_bag_jaccard |
| --- | ---: | ---: |
| hard1_best | 0.1372 | 0.1290 |
| hard1_last | 0.2330 | 0.2271 |

For comparison, the earlier `full3` small run on the same repo-derived eval produced:

| model | perm_stability | target_bag_jaccard |
| --- | ---: | ---: |
| full3 | 0.0588 | 0.0523 |

## Capped Three-Emoji Eval

The sampler can clamp `[EOS]` after three generated response tokens with
`--max-response-tokens 3`. This does not rank or compress emojis by semantic
importance; it lets the diffusion model generate the first three response
positions normally, then fixes `[EOS]` and pads the rest of the sequence.

Eval output:

| model | perm_stability | target_bag_jaccard |
| --- | ---: | ---: |
| hard1_best, capped 3 | 0.1389 | 0.1716 |
| hard1_last, capped 3 | 0.4778 | 0.2561 |

Compared with uncapped `hard1_last` (`0.2330` / `0.2271`), the cap substantially
improves permutation stability and slightly improves target overlap. This is
the better demo setting because emoji replies are naturally short and the
thesis concerns order-invariant semantic response, not filling a 64-token
sequence.

## Demo Samples

From `last.ckpt`, 32 diffusion steps:

```text
🎤🎶🌃 -> 🎺🥁🎧💃🎶
🌃🎤🎶 -> 💃🎺🎷🎶💃
🎶🌃🎤 -> 🎸🎤💃🥁💃

📚💯 -> 📚📒📝📝📝
💯📚 -> 🖊️😣📝🤓🤩

☕🚫🌙 -> 🌞🥱💛🍵🌅
🌙☕🚫 -> 🌞🍵🎧💛🌅
🚫🌙☕ -> 🏃🥱💛🍵🌅

🎉🥳👏 -> ✨🎓🎉🥁🕺
👏🎉🥳 -> 🙌🤩🌟🕺🕺
🥳👏🎉 -> 👏✨✨🏆🌫️
```

## Interpretation

The harder run supports the direction of the thesis better than the previous weak run: more capacity, more semantic scaffolding, longer Phase 2 training, and full permutation augmentation improved permutation stability from `0.0588` to `0.2330`.

It is still not a clean proof. The outputs are often semantically adjacent but not consistently stable or target-matching. The strongest claim for the demo is therefore: atomic emoji diffusion can learn more order-invariant behavior under permutation pressure than the weaker baseline, but repo reply data alone is too sparse/noisy to produce polished emoji reasoning. A compelling hackathon demo should foreground the permutation eval and show the weak qualitative generations honestly beside the improved stability metrics.
