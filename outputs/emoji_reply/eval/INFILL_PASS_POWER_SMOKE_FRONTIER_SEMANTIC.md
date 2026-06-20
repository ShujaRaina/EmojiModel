# Emoji Infill Pass/Power Curves

## Inputs

- `mdlm_smoke=outputs/emoji_phase2_infill_promptbag_h100_smoke/eval/original_promptbag_val_infill_best_s32.jsonl`
- `gpt55=outputs/emoji_reply/eval/openrouter_infill_gpt55_40/openrouter_infill_scored.jsonl`
- `gemini31_flash_lite=outputs/emoji_reply/eval/openrouter_infill_gemini31_flash_lite_40/openrouter_infill_scored.jsonl`

## Curves

| model | pattern | k | problems | pass_at_k_exact_bag | power_at_k_benchmark_score | power_at_k_bag_f1 | pass_at_k_semantic | power_at_k_semantic_cosine | pass_at_k_semantic_distance | power_at_k_semantic_distance | pass_at_k_embedding_distance | power_at_k_embedding_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini31_flash_lite | all | 1 | 120 | 0.0333 | 0.1935 | 0.1167 | 0.4750 | 0.5863 | 0.4833 | 0.2870 | 0.4750 | 0.8604 |
| gemini31_flash_lite | prefix | 1 | 40 | 0.0000 | 0.1544 | 0.0708 | 0.4000 | 0.5395 | 0.4250 | 0.3091 | 0.4000 | 0.9231 |
| gemini31_flash_lite | suffix | 1 | 40 | 0.0750 | 0.2404 | 0.1708 | 0.5250 | 0.6251 | 0.5250 | 0.2662 | 0.5250 | 0.8005 |
| gemini31_flash_lite | scattered | 1 | 40 | 0.0250 | 0.1857 | 0.1083 | 0.5000 | 0.5943 | 0.5000 | 0.2857 | 0.5000 | 0.8576 |
| gpt55 | all | 1 | 120 | 0.0083 | 0.1818 | 0.1056 | 0.4750 | 0.5815 | 0.4833 | 0.2903 | 0.4750 | 0.8687 |
| gpt55 | prefix | 1 | 40 | 0.0000 | 0.1568 | 0.0750 | 0.4250 | 0.5308 | 0.4250 | 0.3104 | 0.4250 | 0.9226 |
| gpt55 | suffix | 1 | 40 | 0.0250 | 0.2026 | 0.1292 | 0.5000 | 0.6277 | 0.5250 | 0.2701 | 0.5000 | 0.8131 |
| gpt55 | scattered | 1 | 40 | 0.0000 | 0.1859 | 0.1125 | 0.5000 | 0.5860 | 0.5000 | 0.2903 | 0.5000 | 0.8705 |
| mdlm_smoke | all | 1 | 729 | 0.0219 | 0.2206 | 0.1541 | 0.5364 | 0.6118 | 0.5569 | 0.2755 | 0.5405 | 0.8300 |
| mdlm_smoke | prefix | 1 | 243 | 0.0165 | 0.2201 | 0.1550 | 0.5226 | 0.6152 | 0.5432 | 0.2756 | 0.5309 | 0.8327 |
| mdlm_smoke | suffix | 1 | 243 | 0.0288 | 0.2215 | 0.1536 | 0.5802 | 0.6242 | 0.5885 | 0.2689 | 0.5802 | 0.8092 |
| mdlm_smoke | scattered | 1 | 243 | 0.0206 | 0.2201 | 0.1536 | 0.5062 | 0.5961 | 0.5391 | 0.2818 | 0.5103 | 0.8481 |
