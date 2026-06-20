# OpenRouter Emoji Eval

- Benchmark: `data/emoji_reply/benchmark.jsonl`
- Prompt mode: `emoji_constrained`
- Prompt variants: `1`
- Samples per problem: `7`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | problems | samples | first_benchmark_score | best7_benchmark_score | pass_at_7_exact_bag | best7_bag_jaccard | permutation_stability | invalid_text_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm_latest | 126 | 882 | 0.3477 | 0.4646 | 0.0000 | 0.3531 | nan | 0.0000 |
| qwen_latest | 126 | 882 | 0.3639 | 0.4593 | 0.0079 | 0.3483 | nan | 0.0000 |
| opus48 | 126 | 882 | 0.3748 | 0.4516 | 0.0079 | 0.3330 | nan | 0.0010 |
| gpt55 | 126 | 882 | 0.3874 | 0.4364 | 0.0159 | 0.3241 | nan | 0.0000 |
| gemini31_flash_lite | 126 | 882 | 0.3451 | 0.4179 | 0.0000 | 0.3025 | nan | 0.0000 |

## Pass / Power Curves

| model | k | pass_at_k_exact_bag | pass_power_at_k_benchmark_score | best_bag_jaccard_at_k |
| --- | --- | --- | --- | --- |
| gemini31_flash_lite | 1 | 0.0000 | 0.3451 | 0.2306 |
| gemini31_flash_lite | 2 | 0.0000 | 0.3708 | 0.2541 |
| gemini31_flash_lite | 3 | 0.0000 | 0.3972 | 0.2802 |
| gemini31_flash_lite | 4 | 0.0000 | 0.4036 | 0.2882 |
| gemini31_flash_lite | 5 | 0.0000 | 0.4096 | 0.2944 |
| gemini31_flash_lite | 6 | 0.0000 | 0.4143 | 0.2991 |
| gemini31_flash_lite | 7 | 0.0000 | 0.4179 | 0.3029 |
| glm_latest | 1 | 0.0000 | 0.3477 | 0.2342 |
| glm_latest | 2 | 0.0000 | 0.3976 | 0.2831 |
| glm_latest | 3 | 0.0000 | 0.4207 | 0.3062 |
| glm_latest | 4 | 0.0000 | 0.4392 | 0.3252 |
| glm_latest | 5 | 0.0000 | 0.4500 | 0.3373 |
| glm_latest | 6 | 0.0000 | 0.4573 | 0.3460 |
| glm_latest | 7 | 0.0000 | 0.4646 | 0.3531 |
| gpt55 | 1 | 0.0079 | 0.3874 | 0.2750 |
| gpt55 | 2 | 0.0079 | 0.4046 | 0.2920 |
| gpt55 | 3 | 0.0079 | 0.4170 | 0.3035 |
| gpt55 | 4 | 0.0079 | 0.4207 | 0.3077 |
| gpt55 | 5 | 0.0079 | 0.4298 | 0.3159 |
| gpt55 | 6 | 0.0159 | 0.4332 | 0.3198 |
| gpt55 | 7 | 0.0159 | 0.4364 | 0.3241 |
| opus48 | 1 | 0.0000 | 0.3748 | 0.2569 |
| opus48 | 2 | 0.0079 | 0.4150 | 0.2977 |
| opus48 | 3 | 0.0079 | 0.4270 | 0.3093 |
| opus48 | 4 | 0.0079 | 0.4397 | 0.3208 |
| opus48 | 5 | 0.0079 | 0.4457 | 0.3263 |
| opus48 | 6 | 0.0079 | 0.4511 | 0.3322 |
| opus48 | 7 | 0.0079 | 0.4516 | 0.3330 |
| qwen_latest | 1 | 0.0000 | 0.3639 | 0.2514 |
| qwen_latest | 2 | 0.0079 | 0.3968 | 0.2848 |
| qwen_latest | 3 | 0.0079 | 0.4271 | 0.3154 |
| qwen_latest | 4 | 0.0079 | 0.4346 | 0.3233 |
| qwen_latest | 5 | 0.0079 | 0.4429 | 0.3310 |
| qwen_latest | 6 | 0.0079 | 0.4512 | 0.3403 |
| qwen_latest | 7 | 0.0079 | 0.4593 | 0.3483 |
