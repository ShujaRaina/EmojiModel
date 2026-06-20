# OpenRouter Emoji Infill Eval

## Setup

- Data file: `data/emoji_reply/emoji_reply.jsonl`
- Excluded benchmark file: `data/emoji_reply/benchmark.jsonl`
- Split: `prompt_bag` validation partition, validation_size=`0.1`, seed=`42`
- Source rows: `2520`
- Benchmark rows excluded: `126`
- Rows after exclusion: `2394`
- Validation split rows: `243`
- Evaluated examples: `40`
- Patterns: `prefix, suffix, scattered`
- Reveal rule: `revealed_k=max(1, reply_len // 2)`; score only hidden positions.
- Mask seed: `1`
- Samples per item: `1`
- Models: `gemini31_flash_lite=google/gemini-3.1-flash-lite`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | pattern | examples | samples | bag_f1 | bag_precision | bag_recall | exact_bag_match | prediction_len | target_len | invalid_text_ratio | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini31_flash_lite | all | 40 | 120 | 0.1167 | 0.1167 | 0.1167 | 0.0333 | 2.0250 | 2.0250 | 0.0000 | 0.0000 |
| gemini31_flash_lite | prefix | 40 | 40 | 0.0708 | 0.0708 | 0.0708 | 0.0000 | 2.0250 | 2.0250 | 0.0000 | 0.0000 |
| gemini31_flash_lite | scattered | 40 | 40 | 0.1083 | 0.1083 | 0.1083 | 0.0250 | 2.0250 | 2.0250 | 0.0000 | 0.0000 |
| gemini31_flash_lite | suffix | 40 | 40 | 0.1708 | 0.1708 | 0.1708 | 0.0750 | 2.0250 | 2.0250 | 0.0000 | 0.0000 |
