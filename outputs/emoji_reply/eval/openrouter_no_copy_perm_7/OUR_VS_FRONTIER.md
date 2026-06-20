# Our Model vs Frontier No-Copy Permutation Eval

Prompt mode: `emoji_no_copy_reply` for frontier models. The frontier prompt contains only prompt emojis plus a user instruction to produce a new emoji-only reply and avoid copying input emojis. No English message or references are provided.

| model | eval | target_score / best7_score | bag_jaccard | permutation_stability | copy_rate | invalid_text_ratio |
|---|---|---:|---:|---:|---:|---:|
| our hard1_last capped3 | saved heldout prompt-bag | 0.4216 | 0.3488 | 0.4028 | n/a | 0.0000 |
| Claude Opus 4.8 | OpenRouter no-copy, 7 samples, 3 variants | 0.3985 | 0.2865 | 0.4453 | 0.0008 mean / 0.0020 best7 | 0.0020 |

Interpretation: our model is ahead on target overlap in the closest saved comparison, while Opus is slightly higher on permutation stability under the no-copy prompt. This is not yet a perfect apples-to-apples comparison because our row set comes from the saved heldout prompt-bag eval and Opus was run on `data/emoji_reply/benchmark.jsonl`; use this as the current demo comparison until we generate benchmark-format predictions from our checkpoint.
