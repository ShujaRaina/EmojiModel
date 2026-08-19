# Held-Out Order-Effect Runs (frontier models only)

Follow-up to
[`outputs/emoji_two_phase_h100_hard1/eval/ORDER_EFFECT_RESULTS.md`](../outputs/emoji_two_phase_h100_hard1/eval/ORDER_EFFECT_RESULTS.md),
which closed with the caveat *"24 prompt groups; widen to the full held-out set
for tighter estimates."* These are those wider runs.

**Read the caveat in "What this does not show" before citing anything here.**

## What was run

`eval/order_eval.py` against `data/emoji_reply/heldout_validation.jsonl` and
`data/emoji_reply/directional_probe.jsonl`, 25 prompt groups × 3 permutations
(23 for the directional probe), best-of-6 scoring against all human replies.
Metric code and group construction are unchanged from the published run.

| file | configuration |
|---|---|
| `order_eval_heldout_results.jsonl` | held-out set, model reasoning **enabled** (matches the published run) |
| `order_eval_heldout_nothink_results.jsonl` | held-out set, model reasoning **disabled** |
| `order_eval_directional_results.jsonl` | directional probe set, reasoning enabled |

Recall `order_effect = resample_stability − perm_stability`. A value near 0
means the model treats the prompt as an order-free bag; larger positive values
mean it is order-sensitive.

## Held-out set, reasoning enabled

| model | perm_stability | resample_stability | order_effect | best-of-6 |
|---|---:|---:|---:|---:|
| openai/gpt-5.5 | 0.6322 | 0.6917 | 0.0595 | 0.2694 |
| anthropic/claude-opus-4.8 | 0.4347 | 0.5039 | 0.0692 | 0.2602 |
| google/gemini-3.1-pro-preview | 0.5787 | 0.5347 | −0.0440 | 0.0599 |

Against the published 24-group numbers for the same models:

| model | published order_effect | held-out order_effect |
|---|---:|---:|
| openai/gpt-5.5 | −0.0221 | 0.0595 |
| anthropic/claude-opus-4.8 | **0.1780** | **0.0692** |
| google/gemini-3.1-pro-preview | **0.2037** | **−0.0440** |

**This softens the published claim.** The headline finding was that MDLM is
order-invariant while Opus-4.8 (0.178) and Gemini-3.1-Pro (0.204) are
"measurably order-sensitive". On the held-out set Opus drops to 0.069 and
Gemini flips sign to −0.044. All three frontier models look approximately
order-invariant here, and the separation the published table reports does not
reproduce at this sample size.

The most likely reading is that 24 groups was too few and the original frontier
order_effect values were noise-dominated — which is what the original caveat
warned about.

## Held-out set, reasoning disabled

Frontier models were re-run with `reasoning: {enabled: false}` so the
comparison against a non-reasoning diffusion model is like-for-like.

| model | perm_stability | resample_stability | order_effect | best-of-6 |
|---|---:|---:|---:|---:|
| openai/gpt-5.5 | 0.5495 | 0.7111 | 0.1616 | 0.2738 |
| anthropic/claude-opus-4.8 | 0.4921 | 0.5019 | 0.0097 | 0.2794 |
| google/gemini-3-flash-preview | 0.5514 | 0.6667 | 0.1153 | 0.2287 |

GPT-5.5's order_effect rises from 0.0595 to 0.1616 with reasoning disabled,
suggesting its reasoning tokens were doing the order-normalisation work rather
than the base model being natively order-invariant. That is a more defensible
version of the project's argument than the published table makes, but it rests
on one model and one run.

Opus-4.8 moves the other way (0.0692 → 0.0097), so this is not a consistent
effect across models.

⚠️ This run substituted `google/gemini-3-flash-preview` for
`gemini-3.1-pro-preview`. That row is **not** comparable to the reasoning-enabled
table above — it is a different model, not the same model with reasoning off.

## Directional probe

| model | perm_stability | resample_stability | order_effect | best-of-6 |
|---|---:|---:|---:|---:|
| openai/gpt-5.5 | 0.4545 | 0.5732 | 0.1187 | 0.1528 |
| anthropic/claude-opus-4.8 | 0.2684 | 0.3636 | 0.0952 | 0.0927 |
| google/gemini-3.1-pro-preview | 0.3182 | 0.4924 | 0.1742 | 0.0208 |

The directional probe is built to contain prompts where order *should* carry
meaning, so a non-zero order_effect is the expected and correct result here.
All three models show one. This is a sanity check on the metric — it confirms
`order_effect` responds to genuine directional structure rather than reading ~0
everywhere.

## What this does not show

**There is no MDLM row in any of these tables.** Every number here is a
frontier baseline. Producing the MDLM side requires the Phase-2 checkpoint,
which was never committed and is no longer available — the H100 instance was
released. So these runs **cannot** be used to make a new MDLM-vs-frontier
claim in either direction.

What they do establish is that the *frontier* half of the published comparison
does not hold up at 25 held-out groups. Before the published order-effect
finding is presented again, the MDLM side needs re-running on this same
held-out set, which means retraining or recovering a checkpoint.

The best-of-6 human-fidelity gap is a separate matter: frontier models score
0.06–0.28 here, against 0.858 published for capped MDLM. That gap is large
enough that it is unlikely to be entirely noise, but it is equally unverified
on this split.
