"""Run OpenRouter frontier baselines on the emoji infilling task.

This mirrors scripts/eval_infilling.py's no-leak data setup for frontier
models: start from data/emoji_reply/emoji_reply.jsonl, remove rows present in
data/emoji_reply/benchmark.jsonl, take the prompt_bag validation partition,
reveal part of each target reply under prefix/suffix/scattered patterns, and
score only the missing emoji positions with bag-F1.
"""
import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

import dataloader
from scripts import emoji_metrics
from scripts import run_openrouter_emoji_eval as openrouter_eval


PATTERNS = ['prefix', 'suffix', 'scattered']
METRIC_KEYS = [
  'bag_f1',
  'bag_precision',
  'bag_recall',
  'bag_jaccard',
  'set_jaccard',
  'length_score',
  'exact_bag_match',
  'exact_sequence_match',
  'invalid_text_ratio',
]


def read_jsonl(path):
  return openrouter_eval.read_jsonl(path)


def write_jsonl(path, rows):
  return openrouter_eval.write_jsonl(path, rows)


def write_csv(path, rows):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else []
  with open(path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def mean(rows, key):
  if not rows:
    return math.nan
  return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def emoji_text(value):
  return ''.join(dataloader.extract_emoji_graphemes(value or ''))


def reply_text(row):
  return (
    row.get('output') or row.get('target_emoji')
    or row.get('response_emoji') or row.get('emoji') or '')


def prompt_text(row):
  return row.get('input') or row.get('prompt_emoji') or ''


def reveal_indices(n, k, pattern, rng):
  idx = list(range(n))
  if pattern == 'prefix':
    return set(idx[:k])
  if pattern == 'suffix':
    return set(idx[n - k:])
  return set(rng.sample(idx, k))


def partition_reply_rows(rows, strategy, validation_size, split_seed,
                         partition):
  if strategy == 'random':
    rng = random.Random(split_seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * validation_size)))
    if partition == 'validation':
      return sorted(shuffled[:valid_count], key=lambda item: item[0])
    if partition == 'train':
      return sorted(shuffled[valid_count:], key=lambda item: item[0])
    if partition == 'all':
      return rows
    raise ValueError('split_partition must be all, train, or validation.')

  if partition == 'all':
    return rows
  keys = [
    dataloader._emoji_reply_split_key(row, strategy)
    for _, row in rows]
  unique_keys = sorted(set(keys))
  if len(unique_keys) <= 1:
    return rows

  def key_sort_value(key):
    import hashlib
    return hashlib.sha1(f'{split_seed}:{key}'.encode('utf-8')).hexdigest()

  shuffled_keys = sorted(unique_keys, key=key_sort_value)
  holdout_count = max(1, int(round(len(shuffled_keys) * validation_size)))
  holdout_count = min(holdout_count, len(shuffled_keys) - 1)
  holdout_keys = set(shuffled_keys[:holdout_count])
  if partition == 'validation':
    return [
      item for item, key in zip(rows, keys)
      if key in holdout_keys]
  if partition == 'train':
    return [
      item for item, key in zip(rows, keys)
      if key not in holdout_keys]
  raise ValueError('split_partition must be all, train, or validation.')


def load_source_examples(args):
  source_rows = [(i, row) for i, row in enumerate(read_jsonl(args.data_file))]
  benchmark_rows = read_jsonl(args.exclude_file)
  benchmark_fingerprints = {
    dataloader._emoji_reply_row_fingerprint(row)
    for row in benchmark_rows}
  filtered_rows = [
    item for item in source_rows
    if dataloader._emoji_reply_row_fingerprint(item[1])
    not in benchmark_fingerprints
  ]
  split_rows = partition_reply_rows(
    filtered_rows,
    args.split_strategy,
    args.split_validation_size,
    args.split_seed,
    args.split_partition)

  examples = []
  for source_index, row in split_rows:
    prompt = emoji_text(prompt_text(row))
    reply = emoji_text(reply_text(row))
    reply_tokens = emoji_metrics.emoji_tokens(reply)
    if len(prompt) < 1 or len(reply_tokens) < 2:
      continue
    examples.append({
      'example_id': make_example_id(source_index, row),
      'source_index': source_index,
      'instruction': row.get('instruction') or row.get('text') or '',
      'prompt_emoji': prompt,
      'reply': ''.join(reply_tokens),
      'reply_tokens': reply_tokens,
      'topic': row.get('topic', ''),
    })
    if args.limit and len(examples) >= args.limit:
      break
  return {
    'source_rows': len(source_rows),
    'excluded_benchmark_rows': len(benchmark_rows),
    'post_exclude_rows': len(filtered_rows),
    'split_rows': len(split_rows),
    'examples': examples,
  }


def make_example_id(source_index, row):
  fingerprint = dataloader._emoji_reply_row_fingerprint(row)[:8]
  return f'emoji_reply_val_{source_index:05d}_{fingerprint}'


def build_infill_items(examples, args):
  rng = random.Random(args.mask_seed)
  items = []
  for example in examples:
    reply_tokens = example['reply_tokens']
    reply_len = len(reply_tokens)
    revealed_k = max(1, reply_len // 2)
    for pattern in PATTERNS:
      revealed = reveal_indices(reply_len, revealed_k, pattern, rng)
      masked_positions = [
        idx for idx in range(reply_len)
        if idx not in revealed]
      masked_truth_tokens = [
        reply_tokens[idx] for idx in masked_positions]
      visible_slots = [
        reply_tokens[idx] if idx in revealed else '[MASK]'
        for idx in range(reply_len)]
      items.append({
        **{key: value for key, value in example.items()
           if key != 'reply_tokens'},
        'pattern': pattern,
        'reply_len': reply_len,
        'revealed_k': revealed_k,
        'revealed_positions': sorted(revealed),
        'masked_positions': masked_positions,
        'masked_count': len(masked_positions),
        'masked_truth': ''.join(masked_truth_tokens),
        'visible_slots': visible_slots,
        'visible_reply': ' '.join(visible_slots),
      })
  return items


def make_prompt(item):
  numbered_slots = '\n'.join(
    f'{index + 1}. {slot}'
    for index, slot in enumerate(item['visible_slots']))
  missing_positions = ', '.join(
    str(index + 1) for index in item['masked_positions'])
  return (
    'Fill the missing emoji positions in an emoji-only reply.\n'
    'Return only the missing emojis in order. No words, no punctuation, '
    'no numbering, no explanation.\n\n'
    f"Original message: {item['instruction']}\n"
    f"Prompt emojis: {item['prompt_emoji']}\n"
    f"Missing positions: {missing_positions}\n"
    f"Missing emoji count: {item['masked_count']}\n"
    'Partially revealed reply slots:\n'
    f'{numbered_slots}\n\n'
    'Missing emojis:')


def task_key(row):
  return (
    row['model'],
    row['example_id'],
    row['pattern'],
    int(row.get('sample_idx', 0)))


def sample_one(api_key, label, model_id, item, sample_idx, args):
  prompt = make_prompt(item)
  error = None
  raw_text = ''
  finish_reason = None
  for attempt in range(args.retries + 1):
    try:
      response = openrouter_eval.openrouter_request(
        api_key, model_id, prompt, args, sample_idx)
      raw_text, finish_reason = openrouter_eval.extract_text(response)
      error = None
      break
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
      error = str(exc)
      if attempt >= args.retries:
        break
      time.sleep(args.retry_sleep * (attempt + 1))
  generated = emoji_metrics.emoji_text(raw_text)
  return {
    'model': label,
    'model_id': model_id,
    'example_id': item['example_id'],
    'source_index': item['source_index'],
    'pattern': item['pattern'],
    'sample_idx': sample_idx,
    'instruction': item['instruction'],
    'topic': item['topic'],
    'prompt_emoji': item['prompt_emoji'],
    'reply': item['reply'],
    'visible_reply': item['visible_reply'],
    'visible_slots': item['visible_slots'],
    'revealed_positions': item['revealed_positions'],
    'masked_positions': item['masked_positions'],
    'revealed_k': item['revealed_k'],
    'reply_len': item['reply_len'],
    'masked_truth': item['masked_truth'],
    'generated_emoji': generated,
    'raw_prediction': raw_text,
    'finish_reason': finish_reason,
    'error': error,
  }


def generate_predictions(items, models, args):
  api_key = os.environ.get(args.api_key_env)
  if not api_key:
    raise SystemExit(
      f'Missing {args.api_key_env}. Set it to an OpenRouter API key.')

  raw_path = os.path.join(args.out_dir, 'openrouter_infill_raw.jsonl')
  rows = []
  existing = set()
  if args.resume and os.path.exists(raw_path):
    for row in read_jsonl(raw_path):
      rows.append(row)
      existing.add(task_key(row))

  tasks = []
  for label, model_id in models:
    for item_index, item in enumerate(items):
      for sample_idx in range(args.samples):
        key = (label, item['example_id'], item['pattern'], sample_idx)
        if key in existing:
          continue
        tasks.append((label, model_id, item_index, item, sample_idx))

  if not tasks:
    return rows

  completed = 0
  os.makedirs(args.out_dir, exist_ok=True)
  with open(raw_path, 'a', encoding='utf-8') as f:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)) as executor:
      future_to_task = {
        executor.submit(
          sample_one,
          api_key,
          label,
          model_id,
          item,
          sample_idx,
          args): (label, item_index, item, sample_idx)
        for label, model_id, item_index, item, sample_idx in tasks
      }
      for future in concurrent.futures.as_completed(future_to_task):
        label, item_index, item, sample_idx = future_to_task[future]
        row = future.result()
        rows.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        completed += 1
        if args.verbose:
          print(
            f"{label} {item_index + 1}/{len(items)} "
            f"{item['pattern']} sample {sample_idx + 1}/{args.samples}: "
            f"{row['generated_emoji']}",
            flush=True)
        elif args.progress_interval and (
            completed == 1 or completed % args.progress_interval == 0):
          print(
            f"Completed {completed}/{len(tasks)} new samples "
            f"({len(rows)} total).",
            flush=True)
        if args.sleep:
          time.sleep(args.sleep)
  return rows


def score_samples(rows):
  scored = []
  for row in rows:
    prediction = emoji_metrics.emoji_text(row.get('generated_emoji') or '')
    raw_prediction = row.get('raw_prediction') or prediction
    target = row.get('masked_truth') or ''
    precision, recall, f1 = emoji_metrics.bag_precision_recall_f1(
      prediction, target)
    metrics = emoji_metrics.score_pair(prediction, target)
    metrics.update({
      'bag_precision': precision,
      'bag_recall': recall,
      'bag_f1': f1,
      'invalid_text_ratio': emoji_metrics.invalid_text_ratio(raw_prediction),
      'error_rate': 1.0 if row.get('error') else 0.0,
    })
    scored.append({
      **row,
      'generated_emoji': prediction,
      **metrics,
    })
  return scored


def make_summary(scored):
  summaries = []
  models = sorted(set(row['model'] for row in scored))
  for model in models:
    model_rows = [row for row in scored if row['model'] == model]
    for pattern in ['all'] + PATTERNS:
      rows = (
        model_rows if pattern == 'all'
        else [row for row in model_rows if row['pattern'] == pattern])
      if not rows:
        continue
      summaries.append({
        'model': model,
        'pattern': pattern,
        'examples': len(set(row['example_id'] for row in rows)),
        'samples': len(rows),
        'bag_f1': mean(rows, 'bag_f1'),
        'bag_precision': mean(rows, 'bag_precision'),
        'bag_recall': mean(rows, 'bag_recall'),
        'bag_jaccard': mean(rows, 'bag_jaccard'),
        'exact_bag_match': mean(rows, 'exact_bag_match'),
        'exact_sequence_match': mean(rows, 'exact_sequence_match'),
        'prediction_len': mean(rows, 'prediction_len'),
        'target_len': mean(rows, 'target_len'),
        'invalid_text_ratio': mean(rows, 'invalid_text_ratio'),
        'error_rate': mean(rows, 'error_rate'),
      })
  summaries.sort(key=lambda row: (row['model'], row['pattern'] != 'all',
                                  row['pattern']))
  return summaries


def markdown_table(rows, columns):
  lines = [
    '| ' + ' | '.join(columns) + ' |',
    '| ' + ' | '.join(['---'] * len(columns)) + ' |',
  ]
  for row in rows:
    values = []
    for column in columns:
      value = row[column]
      if isinstance(value, float):
        values.append(f'{value:.4f}')
      else:
        values.append(str(value))
    lines.append('| ' + ' | '.join(values) + ' |')
  return '\n'.join(lines)


def write_report(path, summaries, args, data_info, models):
  columns = [
    'model',
    'pattern',
    'examples',
    'samples',
    'bag_f1',
    'bag_precision',
    'bag_recall',
    'exact_bag_match',
    'prediction_len',
    'target_len',
    'invalid_text_ratio',
    'error_rate',
  ]
  model_text = ', '.join(f'{label}={model_id}' for label, model_id in models)
  with open(path, 'w', encoding='utf-8') as f:
    f.write('# OpenRouter Emoji Infill Eval\n\n')
    f.write('## Setup\n\n')
    f.write(f'- Data file: `{args.data_file}`\n')
    f.write(f'- Excluded benchmark file: `{args.exclude_file}`\n')
    f.write('- Split: `prompt_bag` validation partition, '
            f'validation_size=`{args.split_validation_size}`, '
            f'seed=`{args.split_seed}`\n')
    f.write(f'- Source rows: `{data_info["source_rows"]}`\n')
    f.write(f'- Benchmark rows excluded: '
            f'`{data_info["excluded_benchmark_rows"]}`\n')
    f.write(f'- Rows after exclusion: `{data_info["post_exclude_rows"]}`\n')
    f.write(f'- Validation split rows: `{data_info["split_rows"]}`\n')
    f.write(f'- Evaluated examples: `{len(data_info["examples"])}`\n')
    f.write(f'- Patterns: `{", ".join(PATTERNS)}`\n')
    f.write('- Reveal rule: `revealed_k=max(1, reply_len // 2)`; '
            'score only hidden positions.\n')
    f.write(f'- Mask seed: `{args.mask_seed}`\n')
    f.write(f'- Samples per item: `{args.samples}`\n')
    f.write(f'- Models: `{model_text}`\n')
    f.write('- Thinking request: `reasoning_effort=none`, '
            '`include_reasoning=false`, `reasoning.exclude=true`\n\n')
    f.write('## Summary\n\n')
    f.write(markdown_table(summaries, columns))
    f.write('\n')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-file',
                      default='data/emoji_reply/emoji_reply.jsonl')
  parser.add_argument('--exclude-file',
                      default='data/emoji_reply/benchmark.jsonl')
  parser.add_argument('--out-dir',
                      default='outputs/emoji_reply/eval/openrouter_infill')
  parser.add_argument('--model', action='append',
                      help='Model id or label=model_id. Use "default" for '
                           'the built-in frontier list.')
  parser.add_argument('--limit', type=int, default=40,
                      help='Number of validation examples before patterns.')
  parser.add_argument('--samples', type=int, default=1)
  parser.add_argument('--split-strategy', default='prompt_bag',
                      choices=[
                        'random',
                        'prompt_bag',
                        'response_bag',
                        'prompt_response_bag'])
  parser.add_argument('--split-partition', default='validation',
                      choices=['all', 'train', 'validation'])
  parser.add_argument('--split-validation-size', type=float, default=0.1)
  parser.add_argument('--split-seed', type=int, default=42)
  parser.add_argument('--mask-seed', type=int, default=1)
  parser.add_argument('--temperature', type=float, default=0.0)
  parser.add_argument('--top-p', type=float, default=1.0)
  parser.add_argument('--max-tokens', type=int, default=32)
  parser.add_argument('--seed', type=int, default=20260620)
  parser.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
  parser.add_argument('--openrouter-url',
                      default='https://openrouter.ai/api/v1/chat/completions')
  parser.add_argument('--referer',
                      default='https://github.com/ShujaRaina/EmojiModel')
  parser.add_argument('--title', default='EmojiModel OpenRouter Infill Eval')
  parser.add_argument('--timeout', type=int, default=90)
  parser.add_argument('--retries', type=int, default=2)
  parser.add_argument('--retry-sleep', type=float, default=2.0)
  parser.add_argument('--sleep', type=float, default=0.0)
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--progress-interval', type=int, default=50)
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--score-only', action='store_true',
                      help='Skip API calls and score openrouter_infill_raw.jsonl.')
  parser.add_argument('--prepare-only', action='store_true',
                      help='Write manifest/items and skip API calls/scoring.')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  # Keep compatibility with the shared OpenRouter request helper.
  args.prompt_mode = 'emoji_constrained'

  models = openrouter_eval.parse_models(args.model)
  os.makedirs(args.out_dir, exist_ok=True)
  data_info = load_source_examples(args)
  items = build_infill_items(data_info['examples'], args)

  manifest = {
    'data_file': args.data_file,
    'exclude_file': args.exclude_file,
    'split_strategy': args.split_strategy,
    'split_partition': args.split_partition,
    'split_validation_size': args.split_validation_size,
    'split_seed': args.split_seed,
    'mask_seed': args.mask_seed,
    'limit': args.limit,
    'patterns': PATTERNS,
    'models': [
      {'label': label, 'model_id': model_id}
      for label, model_id in models],
    **{key: value for key, value in data_info.items()
       if key != 'examples'},
    'examples': len(data_info['examples']),
    'items': len(items),
  }
  with open(os.path.join(args.out_dir, 'openrouter_infill_manifest.json'),
            'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
  write_jsonl(os.path.join(args.out_dir, 'openrouter_infill_items.jsonl'),
              items)

  if args.prepare_only:
    print(
      f"Prepared {len(data_info['examples'])} examples and {len(items)} "
      f"infill items in {args.out_dir}")
    return

  raw_path = os.path.join(args.out_dir, 'openrouter_infill_raw.jsonl')
  if args.score_only:
    if not os.path.exists(raw_path):
      raise SystemExit(f'Missing raw samples file: {raw_path}')
    sample_rows = read_jsonl(raw_path)
  else:
    sample_rows = generate_predictions(items, models, args)

  scored = score_samples(sample_rows)
  summaries = make_summary(scored)
  write_jsonl(os.path.join(args.out_dir, 'openrouter_infill_scored.jsonl'),
              scored)
  write_csv(os.path.join(args.out_dir, 'openrouter_infill_summary.csv'),
            summaries)
  write_report(os.path.join(args.out_dir, 'OPENROUTER_INFILL_RESULTS.md'),
               summaries, args, data_info, models)

  print(markdown_table(summaries, [
    'model',
    'pattern',
    'examples',
    'samples',
    'bag_f1',
    'bag_precision',
    'bag_recall',
    'exact_bag_match',
    'invalid_text_ratio',
    'error_rate',
  ]))
  print()
  print(f'Wrote results to {args.out_dir}')


if __name__ == '__main__':
  main()
