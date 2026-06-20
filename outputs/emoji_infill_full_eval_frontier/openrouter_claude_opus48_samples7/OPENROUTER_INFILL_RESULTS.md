# OpenRouter Emoji Infill Eval

## Setup

- Data file: `data/emoji_reply/emoji_reply.jsonl`
- Excluded benchmark file: `data/emoji_reply/benchmark.jsonl`
- Split: `prompt_bag` validation partition, validation_size=`0.1`, seed=`42`
- Evaluated examples: `243`
- Patterns: `prefix`, `suffix`, `scattered`
- Reveal rule: `revealed_k=max(1, reply_len // 2)`; score only hidden positions.
- Mask seed: `1`
- Samples per item: `7`
- Models: `opus48=anthropic/claude-opus-4.8`
- Thinking request: `reasoning_effort=none`, `include_reasoning=false`, `reasoning.exclude=true`

## Summary

| model | pattern | examples | samples | bag_f1 | bag_precision | bag_recall | bag_jaccard | prediction_len | target_len | invalid_text_ratio | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opus48 | all | 243 | 5103 | 0.10683258214122404 | 0.10683258214122404 | 0.10683258214122404 | 0.07498203671043087 | 2.0536939055457575 | 2.0534979423868314 | 0.0005405137680269956 | 0.0 |
| opus48 | prefix | 243 | 1701 | 0.10719184793258862 | 0.10719184793258862 | 0.10719184793258862 | 0.0748187340779935 | 2.0534979423868314 | 2.0534979423868314 | 0.0005458973712941967 | 0.0 |
| opus48 | scattered | 243 | 1701 | 0.10023515579071125 | 0.10023515579071125 | 0.10023515579071125 | 0.0691553987850285 | 2.054673721340388 | 2.0534979423868314 | 0.0006557228779451002 | 0.0 |
| opus48 | suffix | 243 | 1701 | 0.11307074270037261 | 0.11307074270037261 | 0.11307074270037261 | 0.08097197726827364 | 2.052910052910053 | 2.0534979423868314 | 0.00041992105484168975 | 0.0 |
