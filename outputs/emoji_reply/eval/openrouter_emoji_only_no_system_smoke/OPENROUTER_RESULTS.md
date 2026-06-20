# OpenRouter Emoji Eval

- Benchmark: `data/emoji_reply/benchmark.jsonl`
- Prompt mode: `emoji_only`
- Prompt variants: `1`
- Samples per problem: `1`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | problems | samples | first_benchmark_score | best7_benchmark_score | pass_at_7_exact_bag | best7_bag_jaccard | permutation_stability | invalid_text_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt55 | 1 | 1 | 0.4833 | 0.4833 | 0.0000 | 0.3333 | nan | 0.9718 |

## Pass / Power Curves

| model | k | pass_at_k_exact_bag | pass_power_at_k_benchmark_score | best_bag_jaccard_at_k |
| --- | --- | --- | --- | --- |
| gpt55 | 1 | 0.0000 | 0.4833 | 0.3333 |
