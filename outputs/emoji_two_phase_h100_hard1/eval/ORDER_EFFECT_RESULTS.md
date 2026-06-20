# Order-Effect Eval: MDLM vs Frontier Models

Date: 2026-06-20

Compares `hard1_last` against frontier flagships on emoji-reply permutation
behavior, with a **resample control** that separates order-sensitivity from
sampling noise, and **best-of-6** scoring against all human reply variants.

- MDLM: `scripts/eval_order_effect.py` on the `hard1_last` checkpoint, 32
  diffusion steps, 24 groups × 3 permutations.
- Frontier: `eval/order_eval.py` via OpenRouter, temperature 0.7, same 24
  groups × 3 permutations, identical metric code.

## Metrics

- `perm_stability` — mean pairwise multiset-Jaccard among replies to the
  **permuted** prompts (higher = more stable under shuffling).
- `resample_stability` — same, among replies to the **same** prompt (the
  sampling-noise control).
- `order_effect = resample_stability − perm_stability` — isolates the order
  effect from noise. ≈ 0 means the model treats the prompt as an order-free bag.
- `target_bag_jaccard` — vs the single human reply.
- `target_best6_jaccard` — vs the best of all 6 human replies for that prompt.

## Results

| model | perm_stability | resample_stability | order_effect | target_bag_jaccard | target_best6_jaccard |
|---|---:|---:|---:|---:|---:|
| MDLM hard1_last (capped-3) | 0.6139 | 0.5600 | −0.0539 | 0.2994 | 0.8578 |
| MDLM hard1_last (uncapped) | 0.2461 | 0.2902 | 0.0440 | 0.2500 | 0.5531 |
| openai/gpt-5.5 | 0.5855 | 0.5634 | −0.0221 | 0.1010 | 0.2012 |
| anthropic/claude-opus-4.8 | 0.3480 | 0.5261 | 0.1780 | 0.0682 | 0.1609 |
| google/gemini-3.1-pro-preview | 0.4148 | 0.6185 | 0.2037 | 0.0117 | 0.0457 |

Per-sample generations: `order_effect_uncapped.jsonl`,
`order_effect_capped3.jsonl`, `live_generations_capped3.jsonl`. Frontier
per-sample data: `eval/order_eval_results.jsonl`.

## Findings

1. **Order-invariance (order_effect).** MDLM is order-invariant (0.044
   uncapped, −0.054 capped) — on par with the strongest frontier model
   (GPT-5.5, −0.022) and far better than Opus-4.8 (0.178) and Gemini-3.1-Pro
   (0.204), which are measurably order-sensitive. The resample control was
   essential: MDLM's raw uncapped `perm_stability` (0.246) looked weak, but its
   `resample_stability` is also low (0.290) — that instability is sampling
   noise, not order-sensitivity.

2. **Human fidelity (best-of-6).** MDLM scores 0.553 uncapped / 0.858 capped vs
   frontier's 0.046–0.201 — a 3–17× win. The 328M specialist reproduces human
   emoji replies far better than the flagships.

## Caveats

- 24 prompt groups; widen to the full held-out set for tighter estimates.
- MDLM's diffusion sampler is noisier than the frontier models; `order_effect`
  controls for this but raw `perm_stability` is not directly comparable.
- Jaccard zeroes near-miss-but-apt replies; an embedding metric would be fairer.
