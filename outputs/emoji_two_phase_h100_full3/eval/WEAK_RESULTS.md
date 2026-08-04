# Weak Results: Repo Reply Permutation Eval

> **Superseded — read this alongside
> [`ORDER_EFFECT_RESULTS.md`](../../emoji_two_phase_h100_hard1/eval/ORDER_EFFECT_RESULTS.md).**
>
> The `perm_stability` number below has no resample control, so it cannot
> separate order-sensitivity from diffusion sampling noise. The later
> order-effect eval measures both and finds `order_effect ≈ 0` — the model is
> order-invariant, and the low number here reflects sampler variance, not
> order-sensitivity. This file is kept as a record of the earlier measurement;
> do not cite it as the project's result.

Checkpoint:

```text
outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt
```

Eval source:

```text
data/emoji_reply/emoji_reply.jsonl
```

This eval builds permutation groups only from the repo reply data. For each
eligible row, it shuffles the prompt emoji and compares generated reply emoji
as order-invariant bags.

## Metrics

```text
model  perm_stability  target_bag_jaccard
mdlm   0.0588          0.0523
```

Records written:

```text
outputs/emoji_two_phase_h100_full3/eval/reply_permutation_results.jsonl
```

The result is weak. The model emits emoji-only responses, but the replies are
not stable under prompt permutation and they do not closely match the target
reply emoji bags.

## Sample Failures

```text
🎤🎶🌃 -> 😘🎉🥞🌷😍💖💛        target 🎸🎧
🎶🌃🎤 -> ✨🌻🤩✨🎾🎵📖🎁🥂🎉  target 🎸🎧
🌃🎤🎶 -> 🕺💝🌴🥳🙌            target 🎸🎧

📚💯 -> 🥱🔌📖🌜💪              target ⏰😩📚
💯📚 -> 📖✏️😴⌨️🍕              target ⏰😩📚

☕🚫🌙 -> 💪🌻☕💯💐🎓✨🌅💯🌞  target 😌🫖☕
☕🌙🚫 -> 👏💯🍃🍦              target 😌🫖☕
🚫☕🌙 -> 🍩🌅🍣🌊🌞✨🍔☕      target 😌🫖☕
```

## Commands

Demo:

```bash
python scripts/demo_emoji.py \
  --checkpoint outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_full3 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json \
  --data-file data/emoji_reply/emoji_reply.jsonl \
  --steps 32 \
  --device cuda
```

Eval:

```bash
python scripts/eval_permutation_stability.py \
  --checkpoint mdlm=outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_full3 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json \
  --source reply_data \
  --reply-data-file data/emoji_reply/emoji_reply.jsonl \
  --num-groups 24 \
  --permutations-per-group 3 \
  --steps 32 \
  --device cuda \
  --jsonl-out outputs/emoji_two_phase_h100_full3/eval/reply_permutation_results.jsonl
```
