"""Materialize the emoji-reply validation split as a benchmark JSONL."""
import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

import datasets

import dataloader
from scripts import emoji_metrics


def _read_jsonl(path):
  rows = []
  with open(path, 'r', encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      if not line.strip():
        continue
      row = json.loads(line)
      row['_source_index'] = line_no
      rows.append(row)
  return rows


def _row_fingerprint(row):
  payload = {
    'instruction': row.get('instruction') or row.get('text') or '',
    'input': row.get('input') or row.get('prompt_emoji') or '',
    'output': (
      row.get('output')
      or row.get('target_emoji')
      or row.get('response_emoji')
      or row.get('emoji')
      or ''),
    'topic': row.get('topic') or '',
  }
  return hashlib.sha1(
    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
      'utf-8')).hexdigest()


def _prompt_text(row):
  instruction = row.get('instruction') or row.get('text') or ''
  prompt_emoji = row.get('input') or row.get('prompt_emoji') or ''
  return (
    'Reply with only emojis, no words. Match the mood and topic of the '
    'message. Use 2 to 5 emojis.\n'
    f'Message: {instruction}\n'
    f'Prompt emojis: {prompt_emoji}')


def _reference_key(row):
  return (
    row.get('topic') or '',
    emoji_metrics.emoji_text(row.get('input') or row.get('prompt_emoji') or ''))


def build_benchmark(args):
  rows = _read_jsonl(args.data_file)
  raw = datasets.Dataset.from_list(rows)
  data_config = {
    'emoji_split_strategy': args.split_strategy,
    'emoji_validation_size': args.validation_size,
    'emoji_split_seed': args.seed,
  }
  split = dataloader._split_emoji_reply_dataset(raw, data_config)

  references_by_key = collections.defaultdict(list)
  for row in rows:
    output = row.get('output') or row.get('target_emoji') or ''
    output = emoji_metrics.emoji_text(output)
    if output and output not in references_by_key[_reference_key(row)]:
      references_by_key[_reference_key(row)].append(output)

  benchmark_rows = []
  for row in split['test']:
    source_index = int(row['_source_index'])
    prompt_emoji = emoji_metrics.emoji_text(
      row.get('input') or row.get('prompt_emoji') or '')
    target = emoji_metrics.emoji_text(
      row.get('output') or row.get('target_emoji') or row.get('emoji') or '')
    digest = _row_fingerprint(row)
    references = references_by_key.get(_reference_key(row), [])
    if target and target not in references:
      references = references + [target]
    benchmark_rows.append({
      'benchmark_id': f'emoji_reply_bench_{source_index:05d}_{digest[:8]}',
      'split': 'benchmark',
      'source_index': source_index,
      'source_sha1': digest,
      'instruction': row.get('instruction') or row.get('text') or '',
      'prompt_emoji': prompt_emoji,
      'input': prompt_emoji,
      'target_emoji': target,
      'output': target,
      'topic': row.get('topic') or '',
      'references': references,
      'frontier_prompt': _prompt_text(row),
    })
  benchmark_rows.sort(key=lambda item: item['benchmark_id'])
  return benchmark_rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-file',
                      default='data/emoji_reply/emoji_reply.jsonl')
  parser.add_argument('--output',
                      default='data/emoji_reply/benchmark.jsonl')
  parser.add_argument('--split-strategy',
                      choices=[
                        'random',
                        'prompt_bag',
                        'response_bag',
                        'prompt_response_bag'],
                      default='random')
  parser.add_argument('--validation-size', type=float, default=0.05)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()

  args.data_file = os.path.abspath(args.data_file)
  args.output = os.path.abspath(args.output)
  benchmark_rows = build_benchmark(args)
  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
  with open(args.output, 'w', encoding='utf-8') as f:
    for row in benchmark_rows:
      f.write(json.dumps(row, ensure_ascii=False) + '\n')
  print(f'Wrote {len(benchmark_rows)} benchmark rows to {args.output}')


if __name__ == '__main__':
  main()
