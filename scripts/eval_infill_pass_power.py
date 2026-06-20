"""Compute pass@k and power@k curves for emoji infill JSONL outputs."""
import argparse
import collections
import csv
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_permutation_stability as eps
import emoji_metrics


PATTERN_ORDER = ['all', 'prefix', 'suffix', 'scattered']


def semantic_geodesic_distance(cosine):
  return math.acos(max(-1.0, min(1.0, cosine))) / math.pi


def semantic_embedding_distance_from_cosine(cosine):
  return math.sqrt(max(0.0, 2.0 - 2.0 * max(-1.0, min(1.0, cosine))))


def semantic_embedding_distance(scorer, left, right):
  left_embedding = scorer.embed(left)
  right_embedding = scorer.embed(right)
  if left_embedding is None and right_embedding is None:
    return 0.0
  if left_embedding is None or right_embedding is None:
    return 2.0
  return float(torch.linalg.vector_norm(left_embedding - right_embedding).item())


def read_jsonl(path):
  with open(path, encoding='utf-8') as f:
    return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else []
  with open(path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


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


def infer_label(path, row, fallback):
  return fallback or row.get('model') or os.path.splitext(os.path.basename(path))[0]


def row_problem_key(row):
  example_id = row.get('example_id')
  if not example_id:
    example_id = '|'.join([
      row.get('benchmark_id') or '',
      row.get('prompt') or row.get('prompt_emoji') or '',
      row.get('reply') or '',
    ])
  return (example_id, row.get('pattern') or 'unknown')


def normalized_row(path, row, fallback_label, semantic_scorer=None):
  prediction = (
    row.get('generated_emoji') or row.get('infilled')
    or row.get('prediction') or '')
  target = row.get('masked_truth') or ''
  scored = dict(row)
  if 'benchmark_score' not in scored:
    scored.update(emoji_metrics.score_pair(prediction, target))
  if 'bag_f1' not in scored:
    scored['bag_f1'] = emoji_metrics.bag_precision_recall_f1(
      prediction, target)[2]
  if semantic_scorer is not None and 'semantic_target_cosine' not in scored:
    scored['semantic_target_cosine'] = semantic_scorer.similarity(
      prediction, target)
  if semantic_scorer is not None and 'semantic_embedding_distance' not in scored:
    scored['semantic_embedding_distance'] = semantic_embedding_distance(
      semantic_scorer, prediction, target)
  if 'semantic_target_cosine' in scored and 'semantic_geodesic_distance' not in scored:
    scored['semantic_geodesic_distance'] = semantic_geodesic_distance(
      float(scored['semantic_target_cosine']))
  if 'semantic_target_cosine' in scored and 'semantic_embedding_distance' not in scored:
    scored['semantic_embedding_distance'] = (
      semantic_embedding_distance_from_cosine(
        float(scored['semantic_target_cosine'])))
  scored['_model'] = infer_label(path, scored, fallback_label)
  scored['_problem_key'] = row_problem_key(scored)
  scored['_sample_idx'] = int(scored.get('sample_idx', 0))
  return scored


def parse_input(value):
  if '=' in value:
    label, path = value.split('=', 1)
    return label, path
  return None, value


def load_rows(inputs, semantic_scorer=None):
  rows = []
  for value in inputs:
    label, path = parse_input(value)
    for row in read_jsonl(path):
      rows.append(normalized_row(path, row, label, semantic_scorer))
  return rows


def mean(values):
  return sum(values) / len(values) if values else math.nan


def make_curves(rows, max_k=None, semantic_threshold=0.65,
                semantic_distance_threshold=0.28,
                semantic_embedding_distance_threshold=0.84):
  by_model_pattern_problem = collections.defaultdict(lambda: collections.defaultdict(list))
  for row in rows:
    model = row['_model']
    pattern = row.get('pattern') or 'unknown'
    by_model_pattern_problem[(model, pattern)][row['_problem_key']].append(row)
    by_model_pattern_problem[(model, 'all')][row['_problem_key']].append(row)

  curves = []
  for (model, pattern), by_problem in by_model_pattern_problem.items():
    natural_max_k = max(len(problem_rows) for problem_rows in by_problem.values())
    curve_max_k = min(max_k or natural_max_k, natural_max_k)
    for k in range(1, curve_max_k + 1):
      power_score = []
      power_f1 = []
      power_jaccard = []
      power_semantic = []
      power_semantic_distance = []
      power_embedding_distance = []
      pass_semantic = []
      pass_semantic_distance = []
      pass_embedding_distance = []
      has_semantic = any(
        'semantic_target_cosine' in row
        for problem_rows in by_problem.values()
        for row in problem_rows)
      has_semantic_distance = any(
        'semantic_geodesic_distance' in row
        for problem_rows in by_problem.values()
        for row in problem_rows)
      has_embedding_distance = any(
        'semantic_embedding_distance' in row
        for problem_rows in by_problem.values()
        for row in problem_rows)
      for problem_rows in by_problem.values():
        prefix = sorted(
          problem_rows,
          key=lambda row: row['_sample_idx'])[:k]
        power_score.append(max(
          float(row.get('benchmark_score', 0.0)) for row in prefix))
        power_f1.append(max(float(row.get('bag_f1', 0.0)) for row in prefix))
        power_jaccard.append(max(
          float(row.get('bag_jaccard', 0.0)) for row in prefix))
        if has_semantic:
          best_semantic = max(
            float(row.get('semantic_target_cosine', 0.0))
            for row in prefix)
          power_semantic.append(best_semantic)
          pass_semantic.append(float(best_semantic >= semantic_threshold))
        if has_semantic_distance:
          best_distance = min(
            float(row.get('semantic_geodesic_distance', 1.0))
            for row in prefix)
          power_semantic_distance.append(best_distance)
          pass_semantic_distance.append(float(
            best_distance <= semantic_distance_threshold))
        if has_embedding_distance:
          best_embedding_distance = min(
            float(row.get('semantic_embedding_distance', 2.0))
            for row in prefix)
          power_embedding_distance.append(best_embedding_distance)
          pass_embedding_distance.append(float(
            best_embedding_distance <= semantic_embedding_distance_threshold))
      curve = {
        'model': model,
        'pattern': pattern,
        'k': k,
        'problems': len(by_problem),
        'samples': sum(len(problem_rows) for problem_rows in by_problem.values()),
        'power_at_k_benchmark_score': mean(power_score),
        'power_at_k_bag_f1': mean(power_f1),
        'power_at_k_bag_jaccard': mean(power_jaccard),
      }
      if has_semantic:
        curve['semantic_threshold'] = semantic_threshold
        curve['pass_at_k_semantic'] = mean(pass_semantic)
        curve['power_at_k_semantic_cosine'] = mean(power_semantic)
      if has_semantic_distance:
        curve['semantic_distance_threshold'] = semantic_distance_threshold
        curve['pass_at_k_semantic_distance'] = mean(pass_semantic_distance)
        curve['power_at_k_semantic_distance'] = mean(
          power_semantic_distance)
      if has_embedding_distance:
        curve['semantic_embedding_distance_threshold'] = (
          semantic_embedding_distance_threshold)
        curve['pass_at_k_embedding_distance'] = mean(pass_embedding_distance)
        curve['power_at_k_embedding_distance'] = mean(power_embedding_distance)
      curves.append(curve)

  pattern_rank = {pattern: idx for idx, pattern in enumerate(PATTERN_ORDER)}
  curves.sort(
    key=lambda row: (
      row['model'],
      pattern_rank.get(row['pattern'], len(pattern_rank)),
      row['pattern'],
      row['k']))
  return curves


def write_report(path, curves, inputs):
  columns = [
    'model',
    'pattern',
    'k',
    'problems',
    'power_at_k_benchmark_score',
    'power_at_k_bag_f1',
  ]
  if any('power_at_k_semantic_cosine' in row for row in curves):
    columns.extend(['pass_at_k_semantic', 'power_at_k_semantic_cosine'])
  if any('power_at_k_semantic_distance' in row for row in curves):
    columns.extend([
      'pass_at_k_semantic_distance',
      'power_at_k_semantic_distance'])
  if any('power_at_k_embedding_distance' in row for row in curves):
    columns.extend([
      'pass_at_k_embedding_distance',
      'power_at_k_embedding_distance'])
  with open(path, 'w', encoding='utf-8') as f:
    f.write('# Emoji Infill Pass/Power Curves\n\n')
    f.write('## Inputs\n\n')
    for value in inputs:
      f.write(f'- `{value}`\n')
    f.write('\n## Curves\n\n')
    f.write(markdown_table(curves, columns))
    f.write('\n')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', action='append', required=True,
                      help='JSONL path, optionally label=path.')
  parser.add_argument('--out-csv', required=True)
  parser.add_argument('--out-md', required=True)
  parser.add_argument('--max-k', type=int, default=None)
  parser.add_argument('--semantic-table', default=None,
                      help='Optional semantic_table.pt for centered emoji '
                           'embedding cosine against masked truth.')
  parser.add_argument('--semantic-threshold', type=float, default=0.65,
                      help='Cosine threshold used for semantic pass@k.')
  parser.add_argument('--semantic-distance-threshold', type=float,
                      default=0.28,
                      help='Geodesic distance threshold used for semantic '
                           'distance pass@k. Lower is better.')
  parser.add_argument('--semantic-embedding-distance-threshold', type=float,
                      default=0.84,
                      help='L2 embedding distance threshold used for semantic '
                           'embedding-distance pass@k. Lower is better.')
  args = parser.parse_args()

  semantic_scorer = (
    eps.EmojiSemanticScorer(args.semantic_table)
    if args.semantic_table else None)
  rows = load_rows(args.input, semantic_scorer)
  curves = make_curves(
    rows,
    args.max_k,
    args.semantic_threshold,
    args.semantic_distance_threshold,
    args.semantic_embedding_distance_threshold)
  write_csv(args.out_csv, curves)
  write_report(args.out_md, curves, args.input)
  columns = [
    'model',
    'pattern',
    'k',
    'problems',
    'power_at_k_benchmark_score',
    'power_at_k_bag_f1',
  ]
  if any('power_at_k_semantic_cosine' in row for row in curves):
    columns.extend(['pass_at_k_semantic', 'power_at_k_semantic_cosine'])
  if any('power_at_k_semantic_distance' in row for row in curves):
    columns.extend([
      'pass_at_k_semantic_distance',
      'power_at_k_semantic_distance'])
  if any('power_at_k_embedding_distance' in row for row in curves):
    columns.extend([
      'pass_at_k_embedding_distance',
      'power_at_k_embedding_distance'])
  print(markdown_table(curves, columns))
  print(f'Wrote {args.out_csv}')
  print(f'Wrote {args.out_md}')


if __name__ == '__main__':
  main()
