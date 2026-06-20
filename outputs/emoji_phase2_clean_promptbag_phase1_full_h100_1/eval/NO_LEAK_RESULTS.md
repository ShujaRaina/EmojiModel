# No-Leak Validation Results

Run: `emoji_phase2_clean_promptbag_phase1_full_h100_1`

WandB: https://wandb.ai/derektang-the-university-of-chicago/emoji-model/runs/emoji_phase2_clean_promptbag_phase1_full_h100_1

## Validation Setup

- Final benchmark holdout: `data/emoji_reply/benchmark.jsonl`
- Benchmark rows excluded before train/validation split: 126
- Development split: prompt-bag holdout, `validation_size=0.1`, `seed=42`
- Train rows: 2151
- Validation rows: 243
- Unique prompt bags after benchmark removal: 352
- Heldout validation prompt bags: 35
- Train/validation prompt-bag overlap: 0

The Phase 2 model initializes from the Phase 1 semantic checkpoint:

`outputs/emoji_two_phase_h100_hard1/phase1_semantic/checkpoints/last.ckpt`

It does not initialize from an older reply-trained Phase 2 checkpoint. That avoids carrying reply validation examples through the model weights.

Compatibility caveat: this run reuses the fixed atomic emoji vocab cache from the prior H100 run:

`/tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json`

The vocab is treated as a fixed atomic emoji inventory. Training targets and validation targets are split cleanly; the benchmark rows are excluded before the development split.

## Training Choices

- Model: `medium`
- Sequence length: 64
- Steps: 3000
- Batch size: 512 global
- Precision: bf16 mixed
- Permutation augmentation: prompt and response, probability 1.0
- Response target: full response, no hard 3-token truncation
- PAD supervision: disabled

Validation NLL selected `best.ckpt` at the 1000-step checkpoint. Generation metrics on heldout prompt bags do not perfectly track NLL, so checkpoint selection for the demo should use the validation generation metric directly.

## Heldout Prompt-Bag Eval, 3-Token Demo Clamp

Command output: `heldout_promptbag_capped3_semantic_centered_results.jsonl`

| model | semantic target cosine | semantic stability | bag F1 | bag jaccard | avg decoded len | EOS <= 3 | PAD <= 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_best | 0.5571 | 0.6581 | 0.0997 | 0.0597 | 3.000 | 1.000 | 1.000 |
| clean_last | 0.5737 | 0.6388 | 0.1676 | 0.1058 | 2.981 | 1.000 | 1.000 |
| leaky_hard1_last | 0.8582 | 0.8893 | 0.3765 | 0.3066 | 2.981 | 1.000 | 1.000 |
| leaky_capped_promptbag_best | 0.2713 | 0.1892 | 0.0093 | 0.0056 | 2.769 | 1.000 | 1.000 |

`leaky_hard1_last` and `leaky_capped_promptbag_best` are reference-only. They are not valid no-leak optimization targets because they descend from reply-trained Phase 2 checkpoints that predate the prompt-bag validation split.

## Seeded Clean Checkpoint Sweep

Command output: `heldout_promptbag_capped3_checkpoint_sweep_seed1_results.jsonl`

This sweep sets Python and torch/CUDA RNG seeds before diffusion sampling. It is the reproducible validation-selection pass.

| checkpoint | semantic target cosine | semantic stability | bag F1 | bag jaccard | avg decoded len | EOS <= 3 | PAD <= 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s250 | 0.2425 | 0.2119 | 0.0279 | 0.0164 | 2.654 | 1.000 | 1.000 |
| s500 | 0.2594 | 0.2016 | 0.0231 | 0.0134 | 2.962 | 1.000 | 1.000 |
| s750 | 0.4991 | 0.4512 | 0.1211 | 0.0799 | 3.000 | 1.000 | 1.000 |
| s1000 | 0.5956 | 0.6192 | 0.1523 | 0.0970 | 3.000 | 1.000 | 1.000 |
| s1250 | 0.5794 | 0.6212 | 0.1650 | 0.1036 | 2.981 | 1.000 | 1.000 |
| s1500 | 0.5613 | 0.5679 | 0.1551 | 0.1030 | 3.000 | 1.000 | 1.000 |
| s1750 | 0.5371 | 0.5279 | 0.1247 | 0.0778 | 3.000 | 1.000 | 1.000 |
| s2000 | 0.5860 | 0.5728 | 0.1850 | 0.1230 | 3.000 | 1.000 | 1.000 |
| s2250 | 0.5689 | 0.5769 | 0.1753 | 0.1105 | 2.981 | 1.000 | 1.000 |
| s2500 | 0.5113 | 0.5944 | 0.1524 | 0.0984 | 2.962 | 1.000 | 1.000 |
| s2750 | 0.5562 | 0.6097 | 0.1521 | 0.0955 | 2.981 | 1.000 | 1.000 |
| s3000 | 0.5445 | 0.6687 | 0.1241 | 0.0827 | 2.962 | 1.000 | 1.000 |

Selection:

- Best target alignment: `s2000`
- Best permutation stability: `s3000`
- Best balanced thesis checkpoint: `s1000`

`s1000` has the strongest combined semantic target alignment and semantic permutation stability, while preserving good exact bag metrics. That makes it the recommended validation-selected checkpoint for the order-invariant reasoning demo.

## Heldout Prompt-Bag Eval, Uncapped

Command output: `heldout_promptbag_uncapped_semantic_centered_results.jsonl`

| model | semantic target cosine | semantic stability | bag F1 | bag jaccard | avg decoded len | EOS <= 3 | EOS <= 5 | PAD <= 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_best | 0.5398 | 0.5193 | 0.1234 | 0.0725 | 5.000 | 0.096 | 0.846 | 0.000 |
| clean_last | 0.4940 | 0.6018 | 0.1353 | 0.0860 | 4.519 | 0.212 | 0.904 | 0.000 |

Uncapped generation still tends to produce about 4-5 emoji before EOS. For the demo, a 3-token response clamp is still the cleanest presentation choice. The clamp should remain a decoding-time choice, not a PAD-supervised training target.

## Current Conclusion

For the no-leak validation scheme, use:

`outputs/emoji_phase2_clean_promptbag_phase1_full_h100_1/checkpoints/199-1000.ckpt`

This is also `best.ckpt` from the training run. The older leaky checkpoint is much stronger, but it is not admissible evidence for the new validation scheme.
