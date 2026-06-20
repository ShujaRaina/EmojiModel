# OpenRouter Emoji Eval

- Benchmark: `data/emoji_reply/benchmark.jsonl`
- Prompt mode: `text_assisted`
- Prompt variants: `1`
- Samples per problem: `7`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | problems | samples | first_benchmark_score | best7_benchmark_score | pass_at_7_exact_bag | best7_bag_jaccard | permutation_stability | invalid_text_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_latest | 126 | 882 | 0.3804 | 0.4797 | 0.0238 | 0.3672 | nan | 0.0000 |
| glm_latest | 126 | 882 | 0.3751 | 0.4532 | 0.0079 | 0.3380 | nan | 0.0009 |
| gemini31_flash_lite | 126 | 882 | 0.3805 | 0.4442 | 0.0000 | 0.3261 | nan | 0.0000 |
| gpt55 | 126 | 882 | 0.3860 | 0.4385 | 0.0000 | 0.3195 | nan | 0.0000 |
| opus48 | 126 | 882 | 0.3855 | 0.4334 | 0.0000 | 0.3149 | nan | 0.0000 |

## Pass / Power Curves

| model | k | pass_at_k_exact_bag | pass_power_at_k_benchmark_score | best_bag_jaccard_at_k |
| --- | --- | --- | --- | --- |
| gemini31_flash_lite | 1 | 0.0000 | 0.3805 | 0.2622 |
| gemini31_flash_lite | 2 | 0.0000 | 0.4045 | 0.2853 |
| gemini31_flash_lite | 3 | 0.0000 | 0.4164 | 0.2968 |
| gemini31_flash_lite | 4 | 0.0000 | 0.4322 | 0.3127 |
| gemini31_flash_lite | 5 | 0.0000 | 0.4343 | 0.3144 |
| gemini31_flash_lite | 6 | 0.0000 | 0.4373 | 0.3181 |
| gemini31_flash_lite | 7 | 0.0000 | 0.4442 | 0.3261 |
| glm_latest | 1 | 0.0000 | 0.3751 | 0.2590 |
| glm_latest | 2 | 0.0000 | 0.4107 | 0.2953 |
| glm_latest | 3 | 0.0000 | 0.4259 | 0.3111 |
| glm_latest | 4 | 0.0079 | 0.4345 | 0.3197 |
| glm_latest | 5 | 0.0079 | 0.4416 | 0.3271 |
| glm_latest | 6 | 0.0079 | 0.4488 | 0.3342 |
| glm_latest | 7 | 0.0079 | 0.4532 | 0.3382 |
| gpt55 | 1 | 0.0000 | 0.3860 | 0.2689 |
| gpt55 | 2 | 0.0000 | 0.4110 | 0.2927 |
| gpt55 | 3 | 0.0000 | 0.4225 | 0.3029 |
| gpt55 | 4 | 0.0000 | 0.4305 | 0.3118 |
| gpt55 | 5 | 0.0000 | 0.4330 | 0.3140 |
| gpt55 | 6 | 0.0000 | 0.4377 | 0.3186 |
| gpt55 | 7 | 0.0000 | 0.4385 | 0.3195 |
| opus48 | 1 | 0.0000 | 0.3855 | 0.2674 |
| opus48 | 2 | 0.0000 | 0.4124 | 0.2947 |
| opus48 | 3 | 0.0000 | 0.4230 | 0.3056 |
| opus48 | 4 | 0.0000 | 0.4268 | 0.3090 |
| opus48 | 5 | 0.0000 | 0.4285 | 0.3108 |
| opus48 | 6 | 0.0000 | 0.4305 | 0.3124 |
| opus48 | 7 | 0.0000 | 0.4334 | 0.3150 |
| qwen_latest | 1 | 0.0000 | 0.3804 | 0.2628 |
| qwen_latest | 2 | 0.0000 | 0.4100 | 0.2938 |
| qwen_latest | 3 | 0.0079 | 0.4293 | 0.3120 |
| qwen_latest | 4 | 0.0159 | 0.4439 | 0.3273 |
| qwen_latest | 5 | 0.0238 | 0.4652 | 0.3503 |
| qwen_latest | 6 | 0.0238 | 0.4756 | 0.3637 |
| qwen_latest | 7 | 0.0238 | 0.4797 | 0.3672 |
