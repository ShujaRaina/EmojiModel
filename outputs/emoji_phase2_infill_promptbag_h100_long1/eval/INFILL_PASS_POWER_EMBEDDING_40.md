# Emoji Infill Pass/Power Curves

## Inputs

- `long_best_40=outputs/emoji_phase2_infill_promptbag_h100_long1/eval/original_promptbag_val40_infill_best_s32_semantic.jsonl`
- `long_last_40=outputs/emoji_phase2_infill_promptbag_h100_long1/eval/original_promptbag_val40_infill_last_s32_semantic.jsonl`
- `gpt55_40=outputs/emoji_reply/eval/openrouter_infill_gpt55_40/openrouter_infill_scored.jsonl`
- `gemini31_flash_lite_40=outputs/emoji_reply/eval/openrouter_infill_gemini31_flash_lite_40/openrouter_infill_scored.jsonl`

## Curves

| model | pattern | k | problems | pass_at_k_exact_bag | power_at_k_benchmark_score | power_at_k_bag_f1 | pass_at_k_semantic | power_at_k_semantic_cosine | pass_at_k_semantic_distance | power_at_k_semantic_distance | pass_at_k_embedding_distance | power_at_k_embedding_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini31_flash_lite_40 | all | 1 | 120 | 0.0333 | 0.1935 | 0.1167 | 0.4750 | 0.5863 | 0.4833 | 0.2870 | 0.4750 | 0.8604 |
| gemini31_flash_lite_40 | prefix | 1 | 40 | 0.0000 | 0.1544 | 0.0708 | 0.4000 | 0.5395 | 0.4250 | 0.3091 | 0.4000 | 0.9231 |
| gemini31_flash_lite_40 | suffix | 1 | 40 | 0.0750 | 0.2404 | 0.1708 | 0.5250 | 0.6251 | 0.5250 | 0.2662 | 0.5250 | 0.8005 |
| gemini31_flash_lite_40 | scattered | 1 | 40 | 0.0250 | 0.1857 | 0.1083 | 0.5000 | 0.5943 | 0.5000 | 0.2857 | 0.5000 | 0.8576 |
| gpt55_40 | all | 1 | 120 | 0.0083 | 0.1818 | 0.1056 | 0.4750 | 0.5815 | 0.4833 | 0.2903 | 0.4750 | 0.8687 |
| gpt55_40 | prefix | 1 | 40 | 0.0000 | 0.1568 | 0.0750 | 0.4250 | 0.5308 | 0.4250 | 0.3104 | 0.4250 | 0.9226 |
| gpt55_40 | suffix | 1 | 40 | 0.0250 | 0.2026 | 0.1292 | 0.5000 | 0.6277 | 0.5250 | 0.2701 | 0.5000 | 0.8131 |
| gpt55_40 | scattered | 1 | 40 | 0.0000 | 0.1859 | 0.1125 | 0.5000 | 0.5860 | 0.5000 | 0.2903 | 0.5000 | 0.8705 |
| long_best_40 | all | 1 | 120 | 0.0333 | 0.2298 | 0.1639 | 0.5250 | 0.5968 | 0.5417 | 0.2784 | 0.5333 | 0.8316 |
| long_best_40 | prefix | 1 | 40 | 0.0250 | 0.2649 | 0.2125 | 0.5750 | 0.6448 | 0.6000 | 0.2614 | 0.6000 | 0.7873 |
| long_best_40 | suffix | 1 | 40 | 0.0250 | 0.2146 | 0.1458 | 0.5000 | 0.6096 | 0.5000 | 0.2728 | 0.5000 | 0.8149 |
| long_best_40 | scattered | 1 | 40 | 0.0500 | 0.2098 | 0.1333 | 0.5000 | 0.5359 | 0.5250 | 0.3012 | 0.5000 | 0.8926 |
| long_last_40 | all | 1 | 120 | 0.0333 | 0.2328 | 0.1681 | 0.4667 | 0.5778 | 0.4917 | 0.2865 | 0.4667 | 0.8552 |
| long_last_40 | prefix | 1 | 40 | 0.0250 | 0.2334 | 0.1708 | 0.4750 | 0.5850 | 0.5000 | 0.2837 | 0.4750 | 0.8467 |
| long_last_40 | suffix | 1 | 40 | 0.0500 | 0.2247 | 0.1542 | 0.5250 | 0.6181 | 0.5500 | 0.2732 | 0.5250 | 0.8222 |
| long_last_40 | scattered | 1 | 40 | 0.0250 | 0.2402 | 0.1792 | 0.4000 | 0.5303 | 0.4250 | 0.3025 | 0.4000 | 0.8966 |
