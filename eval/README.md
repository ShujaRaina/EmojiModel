# Emoji-diffusion vs. frontier models — Inspect AI eval

An [Inspect AI](https://inspect.aisi.org.uk/) benchmark that measures the
dimension where a **non-autoregressive** emoji model has a genuine structural
edge over autoregressive frontier models.

## The claim (and what we do *not* claim)

An emoji reply is closer to a **set** than a sequence — its meaning is largely
order-invariant. A masked-diffusion model has the matching inductive bias, so
it should reproduce the **human** emoji-usage distribution (order-free,
diverse) better than an AR model that commits to a canonical order and
mode-collapses to a few "safe" emoji.

We **do not** claim to beat frontier models on raw semantic aptness — they are
bigger and broadly smarter. The headline metric is therefore a
**relevance-gated** blend of the dimensions where the structural advantage
lives. Every sub-metric is reported, so wins *and* losses are visible. Present
it honestly: *"we win distributional/structural fidelity to how humans actually
use emoji; frontier wins raw aptness."* That contrast is the interesting
result.

## Metrics

| metric | what it measures | who should win |
|---|---|---|
| `order_symmetry` | `1 − pairwise-order-bias`: is the emoji order free (human-like) or locked to one canonical order (AR failure mode)? | **diffusion** |
| `diversity` | distinct emoji-*sets* produced per prompt across epochs (mode-collapse detector) | **diffusion** |
| `pure_emoji_rate` | fraction of output that is emoji, not leaked words/preamble | **diffusion** |
| `set_f1_mean` | multiset-F1 vs. the 6 human reference replies (order ignored) | frontier ≈ even |
| `relevance_rate` | grader model: is the reply an on-topic reaction? (the gate) | frontier |
| **`human_fidelity`** | **weighted blend (weights in `emoji_task.py:W`)** | **diffusion** |

The order/diversity metrics need multiple draws per prompt — use `--epochs N`.

## Run

```bash
# 0. install + build the held-out split (once)
pip install inspect_ai
python eval/build_split.py            # -> eval/data/emoji_test.jsonl (112 prompts)

# 1. wiring test, no checkpoint / no API key
python eval/run.py --frontier mockllm/model --epochs 2 --limit 4

# 2. real comparison (frontier + grader via OpenRouter)
export OPENROUTER_API_KEY=sk-or-...
python eval/run.py \
    --diffusion outputs/emoji_run/checkpoints/last.ckpt \
    --frontier openrouter/openai/gpt-4o-mini openrouter/anthropic/claude-3.5-haiku \
    --epochs 8 --steps 128 \
    --relevance-grader openrouter/openai/gpt-4o-mini

inspect view                          # browse per-sample logs
```

Or drive each model straight from the Inspect CLI:

```bash
inspect eval eval/emoji_task.py \
    --model emoji_diffusion/outputs/emoji_run/checkpoints/last.ckpt \
    -M steps=128 -M model=tiny-emoji --epochs 8
inspect eval eval/emoji_task.py --model openai/gpt-4o-mini --epochs 8
```

Set `temperature` equal for both sides (default 1.0) so diversity is compared
fairly — don't hand the frontier model temperature 0 to fake mode-collapse.

## Files

- `build_split.py` — held-out test set; each prompt keeps all 6 human replies
  as the reference distribution.
- `emoji_segment.py` — dependency-free emoji grapheme segmenter + classifier.
- `emoji_provider.py` — Inspect model provider wrapping the MDLM checkpoint
  (`restore_model_and_cond_sample`); registers `emoji_diffusion/...`.
- `emoji_task.py` — dataset, solver, scorer, metrics, headline blend.
- `run.py` — runs all models on one task and prints the leaderboard.

## Honest caveats for the writeup

- Emoji order is *mostly* free, not 100% — there's mild residual signal. Scope
  the claim to "predominantly exchangeable."
- The headline weights are a choice; they up-weight the structural dimensions
  *because those are the thesis*. State the weights on the slide.
- Frontier models will out-score on relevance for novel/out-of-distribution
  prompts. Keep the test set in-distribution (casual one-line texts).
