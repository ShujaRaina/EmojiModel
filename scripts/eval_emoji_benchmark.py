"""Score emoji-reply benchmark predictions from local or frontier models.

Predictions JSONL must include `benchmark_id` and one of:
`generated_emoji`, `output`, `response`, `completion`, or `text`.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

from scripts import emoji_metrics


METRIC_KEYS = [
  'benchmark_score',
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


def _read_jsonl(path):
  rows = []
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def _prediction_text(row):
  for key in ['generated_emoji', 'output', 'response', 'completion', 'text']:
    value = row.get(key)
    if value is not None:
      return str(value)
  return ''


def score_predictions(benchmark_rows, prediction_rows, model_name):
  benchmark_by_id = {
    row['benchmark_id']: row
    for row in benchmark_rows}
  predictions_by_id = {
    row['benchmark_id']: row
    for row in prediction_rows
    if row.get('benchmark_id') in benchmark_by_id}

  scored = []
  missing = []
  for benchmark_id, target_row in benchmark_by_id.items():
    pred_row = predictions_by_id.get(benchmark_id)
    if pred_row is None:
      missing.append(benchmark_id)
      prediction = ''
      raw_prediction = ''
    else:
      raw_prediction = _prediction_text(pred_row)
      prediction = emoji_metrics.emoji_text(raw_prediction)
    references = target_row.get('references') or [
      target_row.get('target_emoji') or target_row.get('output') or '']
    metrics = emoji_metrics.best_reference_score(prediction, references)
    metrics['invalid_text_ratio'] = emoji_metrics.invalid_text_ratio(
      raw_prediction)
    scored.append({
      'model': model_name,
      'benchmark_id': benchmark_id,
      'topic': target_row.get('topic', ''),
      'prompt_emoji': target_row.get('prompt_emoji', target_row.get('input', '')),
      'target_emoji': target_row.get('target_emoji', target_row.get('output', '')),
      'generated_emoji': prediction,
      'raw_prediction': raw_prediction,
      **metrics,
    })
  return scored, missing


def _write_jsonl(path, rows):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_csv(path, summaries):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(
      f, fieldnames=['model', 'n', 'missing'] + METRIC_KEYS)
    writer.writeheader()
    for row in summaries:
      writer.writerow(row)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--benchmark',
                      default='data/emoji_reply/benchmark.jsonl')
  parser.add_argument('--predictions', action='append', required=True,
                      help='JSONL path, optionally model_name=path.')
  parser.add_argument('--jsonl-out', default=None)
  parser.add_argument('--csv-out', default=None)
  args = parser.parse_args()

  benchmark_rows = _read_jsonl(args.benchmark)
  all_scored = []
  summaries = []
  for item in args.predictions:
    if '=' in item:
      model_name, path = item.split('=', 1)
    else:
      path = item
      model_name = os.path.splitext(os.path.basename(path))[0]
    prediction_rows = _read_jsonl(path)
    scored, missing = score_predictions(
      benchmark_rows, prediction_rows, model_name)
    all_scored.extend(scored)
    means = emoji_metrics.mean_dict(scored, METRIC_KEYS)
    summary = {
      'model': model_name,
      'n': len(scored),
      'missing': len(missing),
      **means,
    }
    summaries.append(summary)

  print('\t'.join(['model', 'n', 'missing'] + METRIC_KEYS))
  for row in summaries:
    values = [
      row['model'],
      str(row['n']),
      str(row['missing']),
    ] + [f'{row[key]:.4f}' for key in METRIC_KEYS]
    print('\t'.join(values))

  if args.jsonl_out:
    _write_jsonl(args.jsonl_out, all_scored)
  if args.csv_out:
    _write_csv(args.csv_out, summaries)


if __name__ == '__main__':
  main()
