# OpenRouter Emoji Eval

- Benchmark: `data/emoji_reply/benchmark.jsonl`
- Prompt mode: `emoji_only`
- Prompt variants: `1`
- Samples per problem: `7`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | problems | samples | first_benchmark_score | best7_benchmark_score | pass_at_7_exact_bag | best7_bag_jaccard | permutation_stability | invalid_text_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm_latest | 126 | 882 | 0.2462 | 0.4038 | 0.0159 | 0.3081 | nan | 0.9773 |
| qwen_latest | 126 | 882 | 0.2840 | 0.3926 | 0.0000 | 0.2980 | nan | 0.9671 |
| gpt55 | 126 | 882 | 0.3045 | 0.3576 | 0.0000 | 0.2569 | nan | 0.9284 |
| gemini31_flash_lite | 126 | 882 | 0.2637 | 0.3525 | 0.0000 | 0.2534 | nan | 0.9676 |
| opus48 | 126 | 882 | 0.2383 | 0.3418 | 0.0000 | 0.2563 | nan | 0.9693 |

## Pass / Power Curves

| model | k | pass_at_k_exact_bag | pass_power_at_k_benchmark_score | best_bag_jaccard_at_k |
| --- | --- | --- | --- | --- |
| gemini31_flash_lite | 1 | 0.0000 | 0.2637 | 0.1801 |
| gemini31_flash_lite | 2 | 0.0000 | 0.2929 | 0.2041 |
| gemini31_flash_lite | 3 | 0.0000 | 0.3218 | 0.2266 |
| gemini31_flash_lite | 4 | 0.0000 | 0.3314 | 0.2345 |
| gemini31_flash_lite | 5 | 0.0000 | 0.3443 | 0.2462 |
| gemini31_flash_lite | 6 | 0.0000 | 0.3458 | 0.2478 |
| gemini31_flash_lite | 7 | 0.0000 | 0.3525 | 0.2536 |
| glm_latest | 1 | 0.0079 | 0.2462 | 0.1711 |
| glm_latest | 2 | 0.0079 | 0.3245 | 0.2365 |
| glm_latest | 3 | 0.0079 | 0.3501 | 0.2598 |
| glm_latest | 4 | 0.0079 | 0.3640 | 0.2717 |
| glm_latest | 5 | 0.0079 | 0.3855 | 0.2908 |
| glm_latest | 6 | 0.0159 | 0.3959 | 0.3009 |
| glm_latest | 7 | 0.0159 | 0.4038 | 0.3081 |
| gpt55 | 1 | 0.0000 | 0.3045 | 0.2132 |
| gpt55 | 2 | 0.0000 | 0.3406 | 0.2422 |
| gpt55 | 3 | 0.0000 | 0.3459 | 0.2468 |
| gpt55 | 4 | 0.0000 | 0.3468 | 0.2476 |
| gpt55 | 5 | 0.0000 | 0.3499 | 0.2491 |
| gpt55 | 6 | 0.0000 | 0.3536 | 0.2526 |
| gpt55 | 7 | 0.0000 | 0.3576 | 0.2569 |
| opus48 | 1 | 0.0000 | 0.2383 | 0.1709 |
| opus48 | 2 | 0.0000 | 0.2801 | 0.2019 |
| opus48 | 3 | 0.0000 | 0.3015 | 0.2210 |
| opus48 | 4 | 0.0000 | 0.3077 | 0.2253 |
| opus48 | 5 | 0.0000 | 0.3138 | 0.2315 |
| opus48 | 6 | 0.0000 | 0.3281 | 0.2435 |
| opus48 | 7 | 0.0000 | 0.3418 | 0.2567 |
| qwen_latest | 1 | 0.0000 | 0.2840 | 0.1987 |
| qwen_latest | 2 | 0.0000 | 0.3467 | 0.2534 |
| qwen_latest | 3 | 0.0000 | 0.3745 | 0.2792 |
| qwen_latest | 4 | 0.0000 | 0.3821 | 0.2865 |
| qwen_latest | 5 | 0.0000 | 0.3857 | 0.2910 |
| qwen_latest | 6 | 0.0000 | 0.3886 | 0.2943 |
| qwen_latest | 7 | 0.0000 | 0.3926 | 0.2980 |
