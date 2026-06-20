# Emoji Infill Full Eval - Smoke Best

Full held-out infill evaluation for the current best local checkpoint.

## Files

- `predictions.jsonl`: one row per model sample for each validation example and reveal pattern.
- `summary_pass_power.csv`: tidy pass@k / power@k metrics for k=1..7.
- `SUMMARY_PASS_POWER.md`: Markdown rendering of the same summary table.
- `metadata.json`: run configuration and dataset pointers.

## Evaluation Setup

- Task: emoji reply infilling.
- Source data: `data/emoji_reply/emoji_reply.jsonl`.
- Excluded rows: `data/emoji_reply/benchmark.jsonl`.
- Split: prompt-bag validation, validation size `0.1`, seed `42`.
- Examples: `243`.
- Reveal patterns: `prefix`, `suffix`, `scattered`.
- Samples per problem: `7`.
- Total prediction rows: `5103`.
- Checkpoint: `outputs/emoji_phase2_infill_promptbag_h100_smoke/checkpoints/best.ckpt`.

## Prediction Row Schema

- `model`: model label.
- `example_id`: held-out validation example id.
- `sample_idx`: sample index for pass@k.
- `pattern`: reveal pattern, one of `prefix`, `suffix`, `scattered`.
- `prompt`: emoji prompt.
- `reply`: full target reply.
- `revealed_k`: number of target reply emojis revealed.
- `reply_len`: full target reply length.
- `infilled`: model prediction for hidden target positions.
- `masked_truth`: true hidden target positions.
- `benchmark_score`, `bag_f1`, `bag_precision`, `bag_recall`, `bag_jaccard`: overlap/blended metrics.
- `semantic_target_cosine`: embedding cosine similarity; higher is better.
- `semantic_geodesic_distance`: cosine-derived distance; lower is better.
- `semantic_embedding_distance`: L2 distance in normalized semantic embedding space; lower is better.

