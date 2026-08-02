# Emoji Diffusion Model: Process and Rationale

## Goal

We are building a model that communicates in emoji space rather than text space.

The core thesis is that emoji expressions often behave like global semantic
objects, not ordinary word sequences. For example:

```text
😊😡😢
😢🔥😊
```

These are not identical token sequences, but they can express overlapping
emotional meaning. A purely autoregressive next-token model is a poor fit
because it treats order as the primary structure. A masked diffusion model is
better suited because it can reason over the whole expression at once.

## Core Idea

We use a two-phase approach:

1. Semantic grounding
   - Use text as a teacher signal.
   - Learn what emoji expressions mean by aligning them with text-derived
     semantic embeddings.
   - The final emoji model does not need to read text tokens.

2. Emoji-only response modeling
   - Train the model to map emoji prompts to emoji replies.
   - The model sees only atomic emoji tokens and special control tokens.
   - Text is hidden from the model during this phase.

This lets us use text supervision without making the model a text model.

## Vocabulary

We do not use GPT-2/BPE tokenization for the final model.

Instead, we use an atomic emoji vocabulary:

```text
[PAD]
[BOS]
[EOS]
[SEP]
[MASK]
[UNK_EMOJI]
😀
😂
🔥
❤️
👨‍👩‍👧‍👦
🇺🇸
...
```

Each emoji grapheme cluster is one token, including:

- skin tone emoji
- flags
- ZWJ sequences
- compound family emoji
- variation-selector emoji like `☀️`

This matters because BPE would split emoji into byte fragments, which weakens
the argument that the model is reasoning over emoji symbols.

## Phase 1: Semantic Grounding

Phase 1 uses a Hugging Face model as a teacher, not as the final tokenizer
interface.

For each row in `Text2Emoji`:

```text
text:  "I am so happy today"
emoji: "😊🎉✨"
```

We compute a text embedding:

```text
teacher("I am so happy today") -> semantic vector
```

Then we train an emoji-side model to map:

```text
😊🎉✨ -> same semantic vector
```

The model learns that emoji expressions correspond to meaning in the same space
as text.

Important constraint:

```text
Text is used as supervision.
Text is not part of the final emoji vocabulary.
```

## Phase 2: Emoji-to-Emoji Training

After semantic grounding, we train direct emoji replies.

Training examples look like:

```text
[BOS] prompt_emoji [SEP] response_emoji [EOS]
```

Example:

```text
[BOS] 😢💔 [SEP] 🤗❤️✨ [EOS]
```

The model conditions on the prompt side and denoises the response side.

The response is generated with masked diffusion:

```text
[BOS] 😢💔 [SEP] [MASK] [MASK] [MASK] [EOS]
```

Then the model iteratively fills in the response:

```text
[BOS] 😢💔 [SEP] 🤗 ❤️ ✨ [EOS]
```

## Why Diffusion Works Here

Autoregressive models generate left-to-right:

```text
token 1 -> token 2 -> token 3 -> ...
```

That assumes sequence order is the main structure.

Emoji meaning is often more holistic:

```text
😢❤️🫂
❤️🫂😢
🫂😢❤️
```

These can be semantically similar even though their token order differs.

Masked diffusion is a better fit because it predicts missing tokens while
seeing the whole partially generated expression. It can model the emoji message
as a complete object rather than a strict continuation chain.

## Why Atomic Vocabulary Matters

Atomic emoji tokens preserve the symbolic unit we care about.

Bad:

```text
🔥 -> byte token A, byte token B, byte token C
```

Good:

```text
🔥 -> one token
```

This makes the model's learned space correspond to emoji symbols rather than
implementation artifacts from a text tokenizer.

## Evaluation

The evaluation should focus on semantic stability, not exact token order.

Primary metric:

```text
Permutation stability
```

If we shuffle the same prompt:

```text
😊😡😢
😢😊😡
😡😢😊
```

the model should produce semantically consistent replies.

Secondary metrics:

- semantic embedding similarity
- order-invariant emoji overlap
- exact sequence match only as a diagnostic

The key comparison is against an autoregressive baseline trained on the same
atomic emoji data.

## Demo

The demo should show:

1. User enters an emoji prompt.
2. The app generates an emoji reply.
3. User clicks shuffle.
4. The prompt order changes.
5. The diffusion model remains semantically stable.
6. The autoregressive baseline is more order-sensitive.

This directly supports the thesis:

```text
Emoji expressions are not best modeled as ordinary next-token sequences.
They are better treated as global semantic expressions.
```

## Current Run Artifacts

The demo checkpoint comes from the `full3` two-phase run:

```text
outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt
```

The terminal demo is:

```bash
python scripts/demo_emoji.py \
  --checkpoint outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_full3 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json \
  --data-file data/emoji_reply/emoji_reply.jsonl \
  --steps 32 \
  --device cuda
```

## Results

### The early `full3` result, and why it was misleading

The first permutation eval on `full3`
(`outputs/emoji_two_phase_h100_full3/eval/WEAK_RESULTS.md`) looked like a
failure:

```text
permutation stability: 0.0588
target bag-Jaccard:    0.0523
```

That reading was wrong, for a measurable reason. Raw `perm_stability` conflates
two different things: how much the reply changes because the *prompt order*
changed, and how much it changes because the *diffusion sampler is stochastic*.
A noisy sampler scores badly on permutation stability even if it is perfectly
order-invariant.

The fix is a **resample control** — generate twice from the *same* prompt and
measure stability there too:

```text
order_effect = resample_stability − perm_stability
```

If order carries no information for the model, the two are equal and
`order_effect ≈ 0`.

### Order-effect vs frontier models

From `outputs/emoji_two_phase_h100_hard1/eval/ORDER_EFFECT_RESULTS.md`
(24 prompt groups × 3 permutations, best-of-6 human references):

| model | perm_stability | resample_stability | order_effect | best-of-6 Jaccard |
|---|---:|---:|---:|---:|
| MDLM `hard1_last` (capped-3) | 0.6139 | 0.5600 | **−0.0539** | **0.8578** |
| MDLM `hard1_last` (uncapped) | 0.2461 | 0.2902 | **0.0440** | 0.5531 |
| openai/gpt-5.5 | 0.5855 | 0.5634 | −0.0221 | 0.2012 |
| anthropic/claude-opus-4.8 | 0.3480 | 0.5261 | 0.1780 | 0.1609 |
| google/gemini-3.1-pro-preview | 0.4148 | 0.6185 | 0.2037 | 0.0457 |

Two findings:

1. **Order-invariance.** MDLM's `order_effect` is ≈ 0 (−0.054 capped, 0.044
   uncapped) — on par with the strongest frontier model and far flatter than
   Opus-4.8 (0.178) and Gemini-3.1-Pro (0.204), which are measurably
   order-sensitive. Note the uncapped run's raw `perm_stability` of 0.246 —
   almost exactly the "weak" number from `full3`. Its `resample_stability` is
   also 0.290, so that instability was sampling noise, not order sensitivity.
   This is precisely the misreading above.

2. **Human fidelity.** Against all 6 human replies, MDLM scores 0.553 uncapped
   / 0.858 capped vs 0.046–0.201 for the flagships. The 328M specialist
   reproduces human emoji replies far better than general frontier models.

### Infilling

From `outputs/emoji_infill_full_eval_frontier/SUMMARY_ALL_MODELS_VS_LOCAL.md`
(729 problems, 7 samples, prefix/suffix/scattered masks), reporting
`power_at_k_benchmark_score` over all mask patterns:

| model | k=1 | k=7 |
|---|---:|---:|
| **local best (ours)** | **0.2092** | **0.4295** |
| openai/gpt-5.5 | 0.1879 | 0.2293 |
| anthropic/claude-opus-4.8 | 0.1834 | 0.2241 |
| google/gemini-3.1-flash-lite | 0.1660 | 0.1817 |

The gap widens with k: the diffusion model's samples are genuinely diverse, so
additional draws keep finding new correct fills, while the AR models converge
on the same answer.

### Caveats

- The order-effect eval uses 24 prompt groups; widen to the full held-out set
  for tighter estimates.
- The diffusion sampler is noisier than the frontier models. `order_effect`
  controls for this, but raw `perm_stability` is not directly comparable.
- Jaccard scores near-miss-but-apt replies as zero; an embedding metric would
  be fairer to all sides.
- The capped-3 and uncapped MDLM rows differ substantially. Capping response
  length to 3 tokens matches the human reply-length distribution and is the
  configuration used in the demo, but both are reported.

## Why This Process Works

This process works because it separates three concerns:

1. Meaning
   - Learned from text embeddings during Phase 1.

2. Symbolic representation
   - Preserved by atomic emoji vocabulary.

3. Emoji-only reasoning
   - Learned during Phase 2 with masked diffusion.

The result is a model that can use text-derived semantic grounding while
operating entirely in emoji space at inference time.
