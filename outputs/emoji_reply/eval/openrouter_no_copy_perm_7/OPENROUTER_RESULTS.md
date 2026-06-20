# OpenRouter Emoji Eval

- Benchmark: `data/emoji_reply/benchmark.jsonl`
- Prompt mode: `emoji_no_copy_reply`
- Prompt variants: `3`
- Samples per problem: `7`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | problems | samples | first_benchmark_score | best7_benchmark_score | pass_at_7_exact_bag | best7_bag_jaccard | first_copy_rate | mean_copy_rate | best7_copy_rate | permutation_stability | permutation_stability_n | permutation_stability_status | invalid_text_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opus48 | 126 | 1757 | 0.2914 | 0.3985 | 0.0000 | 0.2865 | 0.0000 | 0.0008 | 0.0020 | 0.4453 | 118 | ok | 0.0020 |

## Pass / Power Curves

| model | k | pass_at_k_exact_bag | pass_power_at_k_benchmark_score | best_bag_jaccard_at_k |
| --- | --- | --- | --- | --- |
| opus48 | 1 | 0.0000 | 0.2914 | 0.1788 |
| opus48 | 2 | 0.0000 | 0.3236 | 0.2093 |
| opus48 | 3 | 0.0000 | 0.3410 | 0.2271 |
| opus48 | 4 | 0.0000 | 0.3538 | 0.2396 |
| opus48 | 5 | 0.0000 | 0.3562 | 0.2421 |
| opus48 | 6 | 0.0000 | 0.3731 | 0.2587 |
| opus48 | 7 | 0.0000 | 0.3773 | 0.2637 |
