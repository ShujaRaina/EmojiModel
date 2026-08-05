# Order-Effect Eval: MDLM vs Frontier Models

> # ⛔ RETRACTED
>
> **Do not cite any number in this file.** It is kept for provenance, not as
> evidence. Six independent problems were verified, any one of the first four
> disqualifying on its own. Full detail in the README's Limitations section.
>
> 1. **The invariance was trained in.** This run used
>    `permutation_augment_prob: 1.0` on both prompt and response
>    (`../phase2_reply/.hydra/config.yaml`). Order-invariance was injected as
>    augmentation, not exhibited by the architecture. An AR model trained
>    identically would show it too.
> 2. **Evaluated on training data with a checkpoint this repo calls invalid.**
>    `--split-partition all` reads the Phase-2 training file.
>    `NO_LEAK_RESULTS.md` labels this checkpoint `leaky_hard1_last`,
>    "reference-only … not valid … not admissible evidence."
> 3. **A baseline that mostly failed, scored as success.**
>    `bag_jaccard("","") == 1.0`, and Gemini-3.1-Pro returned empty in 57% of
>    permutation and 64% of resample samples, inflating its `order_effect`.
>    Filtering GPT-5.5's affected groups flips its sign.
> 4. **The AR comparison is rigged and was never run.** The AR sampler decodes
>    with `argmax`, so `resample_stability ≡ 1.0` and
>    `order_effect ≡ 1 − perm_stability` by construction. No AR run exists.
> 5. **Everything is inside the noise.** Four of five bootstrap CIs contain
>    zero; MDLM's two configs give opposite signs.
> 6. **The metric is not scale-free, and the checkpoint was picked using it.**
>    Normalising to `1 − perm/resample` makes MDLM uncapped (+0.152) *more*
>    order-sensitive than GPT-5.5 (−0.039). `HARD_RESULTS.md` records choosing
>    a checkpoint with worse validation NLL "because it showed better
>    permutation stability."
>
> The infilling results elsewhere in `outputs/` are unaffected — those use a
> proper prompt-bag holdout and remain the project's defensible finding.

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

2. **Best-of-6 fidelity.** MDLM scores 0.553 uncapped / 0.858 capped vs
   frontier's 0.046–0.201. ⚠️ This is **not** a human-fidelity result. The six
   references are six random draws from a hand-authored per-topic emoji pool,
   which our model trained on and the frontier models have never seen. The gap
   mostly measures that asymmetry, not better emoji reasoning. Reporting it as
   "reproduces human emoji replies better than the flagships" was wrong.

## Caveats

- 24 prompt groups; widen to the full held-out set for tighter estimates.
- MDLM's diffusion sampler is noisier than the frontier models; `order_effect`
  controls for this but raw `perm_stability` is not directly comparable.
- Jaccard zeroes near-miss-but-apt replies; an embedding metric would be fairer.
