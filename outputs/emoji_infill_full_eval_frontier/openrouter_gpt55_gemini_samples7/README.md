# OpenRouter Frontier Emoji Infill Full Eval

Full held-out OpenRouter frontier baseline evaluation for the emoji reply
infilling task.

## Files

- `openrouter_infill_raw.jsonl`: raw OpenRouter responses.
- `openrouter_infill_scored.jsonl`: scored rows for both frontier models.
- `gpt55_predictions.jsonl`: scored rows for `openai/gpt-5.5`.
- `gemini31_flash_lite_predictions.jsonl`: scored rows for `google/gemini-3.1-flash-lite`.
- `summary_pass_power.csv`: frontier-only pass@k / power@k summary.
- `SUMMARY_PASS_POWER.md`: Markdown rendering of the frontier-only summary.
- `summary_vs_local_smoke.csv`: local `smoke_best` vs frontier summary.
- `SUMMARY_VS_LOCAL_SMOKE.md`: Markdown rendering of the local-vs-frontier summary.
- `openrouter_infill_manifest.json`: OpenRouter eval manifest.
- `openrouter_infill_items.jsonl`: infill prompts/items before sampling.
- `OPENROUTER_INFILL_RESULTS.md`: basic OpenRouter bag-F1 summary.
- `metadata.json`: compact machine-readable metadata.

## Evaluation Setup

- Task: emoji reply infilling.
- Source data: `data/emoji_reply/emoji_reply.jsonl`.
- Excluded rows: `data/emoji_reply/benchmark.jsonl`.
- Split: prompt-bag validation, validation size `0.1`, seed `42`.
- Examples: `243`.
- Reveal patterns: `prefix`, `suffix`, `scattered`.
- Samples per problem: `7`.
- Models: `openai/gpt-5.5`, `google/gemini-3.1-flash-lite`.
- Total scored rows: `10206`.

Primary semantic metrics are computed from
`outputs/emoji_two_phase_h100_hard1/semantic_table.pt`.

