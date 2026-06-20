# Heldout Prompt-Bag Phase 2 Results

Date: 2026-06-20

## Runs

- In-distribution capped Phase 2:
  - Output: `outputs/emoji_phase2_capped3_pad2_h100_1`
  - Init: `outputs/emoji_two_phase_h100_hard1/phase2_reply/checkpoints/last.ckpt`
  - Config: `emoji_max_response_tokens=3`, `emoji_supervised_pad_tokens=2`, random split, 500 steps
  - Best validation NLL: 2.40227 at step 400
  - WandB: `emoji_phase2_capped3_pad2_h100_1`
- Heldout prompt-bag capped Phase 2:
  - Output: `outputs/emoji_phase2_capped3_pad2_promptbag_h100_1`
  - Init: `outputs/emoji_two_phase_h100_hard1/phase2_reply/checkpoints/last.ckpt`
  - Config: `emoji_split_strategy=prompt_bag`, `emoji_validation_size=0.1`, `emoji_split_seed=42`, `emoji_max_response_tokens=3`, `emoji_supervised_pad_tokens=2`, 500 steps
  - Split: 2262 train rows, 258 validation rows, 352 unique bags, 35 held out
  - Best validation NLL: 2.59824 at step 350
  - WandB: `emoji_phase2_capped3_pad2_promptbag_h100_1`

## Centered Semantic Eval

Eval source: reply data, heldout `prompt_bag` validation partition, 24 groups, 3 permutations per group, 32 diffusion steps.

Semantic scorer: per-emoji vectors are built offline from `semantic_table.pt`, then centered and normalized before cosine scoring. This avoids the saturated BGE cosine behavior from raw embedding space.

### Uncapped Decoding

Artifact: `heldout_promptbag_uncapped_semantic_centered_results.jsonl`

| model | perm_stability | target_score | target_bag_f1 | target_bag_jaccard | semantic_stability | semantic_target_cosine |
|---|---:|---:|---:|---:|---:|---:|
| hard1_last | 0.2904 | 0.3005 | 0.2872 | 0.2057 | 0.7991 | 0.7652 |
| capped_random_best | 0.0228 | 0.0789 | 0.0108 | 0.0061 | 0.2298 | 0.2631 |
| promptbag_best | 0.0046 | 0.0879 | 0.0210 | 0.0122 | 0.2925 | 0.2631 |
| promptbag_last | 0.0080 | 0.0909 | 0.0158 | 0.0092 | 0.2277 | 0.2636 |

EOS/PAD summary:

| model | n | avg_len | eos<=3 | eos<=5 | pad<=3 | no_eos | semantic_target |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard1_last | 52 | 4.73 | 0.06 | 1.00 | 0.00 | 0.00 | 0.7681 |
| capped_random_best | 52 | 4.08 | 0.38 | 0.42 | 0.42 | 0.00 | 0.2664 |
| promptbag_best | 52 | 3.13 | 0.27 | 0.35 | 0.48 | 0.00 | 0.2678 |
| promptbag_last | 52 | 3.06 | 0.38 | 0.44 | 0.54 | 0.00 | 0.2741 |

### Capped 3-Token Decoding

Artifact: `heldout_promptbag_capped3_semantic_centered_results.jsonl`

| model | perm_stability | target_score | target_bag_f1 | target_bag_jaccard | semantic_stability | semantic_target_cosine |
|---|---:|---:|---:|---:|---:|---:|
| hard1_last | 0.4028 | 0.4216 | 0.3981 | 0.3488 | 0.8124 | 0.8235 |
| capped_random_best | 0.0187 | 0.0913 | 0.0156 | 0.0091 | 0.2831 | 0.2746 |
| promptbag_best | 0.0028 | 0.0927 | 0.0231 | 0.0139 | 0.1775 | 0.2088 |
| promptbag_last | 0.0000 | 0.0956 | 0.0181 | 0.0106 | 0.2847 | 0.2515 |

EOS/PAD summary:

| model | n | avg_len | eos<=3 | eos<=5 | pad<=3 | no_eos | semantic_target |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard1_last | 52 | 2.98 | 1.00 | 1.00 | 0.00 | 0.00 | 0.8235 |
| capped_random_best | 52 | 2.92 | 1.00 | 1.00 | 0.00 | 0.00 | 0.2768 |
| promptbag_best | 52 | 2.69 | 1.00 | 1.00 | 0.00 | 0.00 | 0.2211 |
| promptbag_last | 52 | 3.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.2613 |

## Conclusion

The heldout prompt-bag split and centered semantic metric are working, but the new capped Phase 2 retrains did not improve heldout generation. The strongest current demo checkpoint remains `hard1_last` with capped 3-token decoding.

The capped retrains learned EOS/PAD mechanics but lost semantic relevance and permutation stability. For the next training-improvement attempt, avoid hard-truncating every target during training; prefer keeping full targets while using decode-time cap, lower LR/fewer finetune steps from `hard1_last`, or add a softer length objective that does not reward early PAD collapse.
