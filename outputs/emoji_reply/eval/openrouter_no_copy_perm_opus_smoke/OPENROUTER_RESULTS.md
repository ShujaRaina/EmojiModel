# OpenRouter Emoji Eval

- Benchmark: `data/emoji_reply/benchmark.jsonl`
- Prompt mode: `emoji_no_copy_reply`
- Prompt variants: `3`
- Samples per problem: `2`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | problems | samples | first_benchmark_score | best7_benchmark_score | pass_at_7_exact_bag | best7_bag_jaccard | first_copy_rate | mean_copy_rate | best7_copy_rate | permutation_stability | permutation_stability_n | permutation_stability_status | invalid_text_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opus48 | 10 | 44 | 0.3714 | 0.4530 | 0.0000 | 0.3383 | 0.0000 | 0.0000 | 0.0000 | 0.5325 | 10 | ok | 0.0000 |

## Pass / Power Curves

| model | k | pass_at_k_exact_bag | pass_power_at_k_benchmark_score | best_bag_jaccard_at_k |
| --- | --- | --- | --- | --- |
| opus48 | 1 | 0.0000 | 0.3714 | 0.2545 |
| opus48 | 2 | 0.0000 | 0.4246 | 0.3117 |
