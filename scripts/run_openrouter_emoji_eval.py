"""Run OpenRouter frontier-model emoji benchmark evals.

This script generates multiple samples per benchmark row, scores each sample
with the same emoji benchmark verifier used by eval_emoji_benchmark.py, and
writes pass@k plus best-of-k score curves for k=1..N.
"""
import argparse
import csv
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

from scripts import emoji_metrics


DEFAULT_MODELS = [
  ('gpt55', 'openai/gpt-5.5'),
  ('opus48', 'anthropic/claude-opus-4.8'),
  # Gemini 3.1 Pro currently mandates reasoning on OpenRouter, so the default
  # thinking-off Gemini entry uses the newest tested Gemini endpoint that
  # accepts reasoning_effort=none.
  ('gemini31_flash_lite', 'google/gemini-3.1-flash-lite'),
  ('qwen_latest', 'qwen/qwen3.7-plus'),
  ('glm_latest', 'z-ai/glm-5.2'),
]

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
  'copy_rate',
]


def read_jsonl(path):
  rows = []
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def write_jsonl(path, rows):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_models(values):
  if not values:
    return DEFAULT_MODELS
  models = []
  for value in values:
    if value == 'default':
      models.extend(DEFAULT_MODELS)
      continue
    if '=' in value:
      label, model_id = value.split('=', 1)
    else:
      model_id = value
      label = (
        model_id.replace('/', '_')
        .replace(':', '_')
        .replace('.', '_')
        .replace('-', '_'))
    models.append((label, model_id))
  return models


def openrouter_request(api_key, model_id, prompt, args, sample_idx):
  messages = [{'role': 'user', 'content': prompt}]
  if args.prompt_mode == 'text_assisted':
    messages.insert(0, {
      'role': 'system',
      'content': (
        'Reply with only emojis. Use 2 to 5 emojis. No words, no '
        'punctuation, no explanation.'),
    })
  payload = {
    'model': model_id,
    'messages': messages,
    'temperature': args.temperature,
    'top_p': args.top_p,
    'max_tokens': args.max_tokens,
    'seed': args.seed + sample_idx,
    'reasoning_effort': 'none',
    'include_reasoning': False,
    'reasoning': {
      'effort': 'none',
      'enabled': False,
      'exclude': True,
    },
  }
  data = json.dumps(payload).encode('utf-8')
  request = urllib.request.Request(
    args.openrouter_url,
    data=data,
    method='POST',
    headers={
      'Authorization': f'Bearer {api_key}',
      'Content-Type': 'application/json',
      'HTTP-Referer': args.referer,
      'X-OpenRouter-Title': args.title,
    })
  with urllib.request.urlopen(request, timeout=args.timeout) as response:
    return json.loads(response.read().decode('utf-8'))


def extract_text(response):
  choices = response.get('choices') or []
  if not choices:
    return '', None
  choice = choices[0]
  message = choice.get('message') or {}
  content = message.get('content') or ''
  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, dict) and item.get('type') == 'text':
        parts.append(item.get('text', ''))
      elif isinstance(item, str):
        parts.append(item)
    content = ''.join(parts)
  return str(content), choice.get('finish_reason')


def _text_assisted_prompt(bench):
  return bench.get('frontier_prompt') or (
    f"Message: {bench.get('instruction', '')}\n"
    f"Prompt emojis: {bench.get('prompt_emoji', bench.get('input', ''))}")


def _emoji_only_variants(bench, count, seed):
  prompt = bench.get('prompt_emoji', bench.get('input', '')) or ''
  tokens = emoji_metrics.emoji_tokens(prompt)
  if not tokens:
    return ['']

  variants = [''.join(tokens)]
  seen = set(variants)
  rng_seed = int(hashlib.sha1(
    f"{seed}:{bench['benchmark_id']}".encode('utf-8')).hexdigest()[:12],
                 16)
  rng = random.Random(rng_seed)
  attempts = 0
  while len(variants) < count and len(tokens) > 1 and attempts < 128:
    attempts += 1
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    candidate = ''.join(shuffled)
    if candidate not in seen:
      variants.append(candidate)
      seen.add(candidate)
  return variants


def _prompt_variants(bench, args):
  prompt_emoji = bench.get('prompt_emoji', bench.get('input', '')) or ''
  if args.prompt_mode == 'text_assisted':
    return [(0, prompt_emoji, _text_assisted_prompt(bench))]

  variants = _emoji_only_variants(bench, args.prompt_variants, args.seed)
  if args.prompt_mode == 'emoji_only':
    return [(idx, variant, variant) for idx, variant in enumerate(variants)]
  if args.prompt_mode == 'emoji_constrained':
    return [
      (
        idx,
        variant,
        'Prompt emojis: '
        f'{variant}\n'
        'Reply with only 2 to 5 emojis. Match the meaning. Do not use words.'
      )
      for idx, variant in enumerate(variants)
    ]
  return [
    (
      idx,
      variant,
      'Emoji message: '
      f'{variant}\n'
      'Write a NEW emoji-only reply, not a copy of the input.\n'
      'Use 2 to 5 emojis.\n'
      'Do not reuse any input emoji unless absolutely necessary.\n'
      'Reply:'
    )
    for idx, variant in enumerate(variants)
  ]


def _sample_one(api_key, label, model_id, bench, prompt, prompt_variant_idx,
                prompt_variant_emoji, sample_idx, args):
  error = None
  raw_text = ''
  finish_reason = None
  for attempt in range(args.retries + 1):
    try:
      response = openrouter_request(
        api_key, model_id, prompt, args, sample_idx)
      raw_text, finish_reason = extract_text(response)
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
    'benchmark_id': bench['benchmark_id'],
    'prompt_mode': args.prompt_mode,
    'prompt_variant_idx': prompt_variant_idx,
    'prompt_variant_emoji': prompt_variant_emoji,
    'sample_idx': sample_idx,
    'generated_emoji': generated,
    'raw_prediction': raw_text,
    'finish_reason': finish_reason,
    'error': error,
  }


def _task_list(benchmark_rows, models, existing, args):
  tasks = []
  for label, model_id in models:
    for bench_index, bench in enumerate(benchmark_rows):
      for prompt_variant_idx, prompt_variant_emoji, prompt in _prompt_variants(
          bench, args):
        for sample_idx in range(args.samples):
          key = (
            label,
            bench['benchmark_id'],
            prompt_variant_idx,
            sample_idx)
          if key in existing:
            continue
          tasks.append((
            label,
            model_id,
            bench_index,
            bench,
            prompt,
            prompt_variant_idx,
            prompt_variant_emoji,
            sample_idx))
  return tasks


def generate_predictions(benchmark_rows, models, args):
  api_key = os.environ.get(args.api_key_env)
  if not api_key:
    raise SystemExit(
      f'Missing {args.api_key_env}. Set it to an OpenRouter API key.')

  rows = []
  raw_path = os.path.join(args.out_dir, 'openrouter_raw_samples.jsonl')
  os.makedirs(args.out_dir, exist_ok=True)
  existing = set()
  if args.resume and os.path.exists(raw_path):
    for row in read_jsonl(raw_path):
      rows.append(row)
      existing.add((
        row['model'],
        row['benchmark_id'],
        int(row.get('prompt_variant_idx', 0)),
        row['sample_idx']))

  tasks = _task_list(benchmark_rows, models, existing, args)
  if not tasks:
    return rows

  completed = 0
  with open(raw_path, 'a', encoding='utf-8') as f:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)) as executor:
      future_to_task = {
        executor.submit(
          _sample_one,
          api_key,
          label,
          model_id,
          bench,
          prompt,
          prompt_variant_idx,
          prompt_variant_emoji,
          sample_idx,
          args): (label, bench_index, prompt_variant_idx, sample_idx)
        for (
          label,
          model_id,
          bench_index,
          bench,
          prompt,
          prompt_variant_idx,
          prompt_variant_emoji,
          sample_idx) in tasks
      }
      for future in concurrent.futures.as_completed(future_to_task):
        label, bench_index, prompt_variant_idx, sample_idx = future_to_task[
          future]
        row = future.result()
        rows.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        completed += 1
        if args.verbose:
          print(
            f"{label} {bench_index + 1}/{len(benchmark_rows)} "
            f"variant {prompt_variant_idx} "
            f"sample {sample_idx + 1}/{args.samples}: "
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


def score_samples(benchmark_rows, sample_rows):
  benchmark_by_id = {
    row['benchmark_id']: row
    for row in benchmark_rows}
  scored = []
  for row in sample_rows:
    bench = benchmark_by_id.get(row.get('benchmark_id'))
    if not bench:
      continue
    prediction = emoji_metrics.emoji_text(row.get('generated_emoji') or '')
    raw_prediction = row.get('raw_prediction') or prediction
    references = bench.get('references') or [
      bench.get('target_emoji') or bench.get('output') or '']
    metrics = emoji_metrics.best_reference_score(prediction, references)
    metrics['invalid_text_ratio'] = emoji_metrics.invalid_text_ratio(
      raw_prediction)
    prompt_variant = row.get(
      'prompt_variant_emoji',
      bench.get('prompt_emoji', bench.get('input', '')))
    metrics['copy_rate'] = copy_rate(prediction, prompt_variant)
    scored.append({
      **row,
      'topic': bench.get('topic', ''),
      'prompt_emoji': bench.get('prompt_emoji', bench.get('input', '')),
      'prompt_variant_emoji': prompt_variant,
      'target_emoji': bench.get('target_emoji', bench.get('output', '')),
      **metrics,
    })
  return scored


def copy_rate(prediction, prompt):
  pred_tokens = emoji_metrics.emoji_tokens(prediction)
  if not pred_tokens:
    return 0.0
  prompt_tokens = set(emoji_metrics.emoji_tokens(prompt))
  return sum(1 for token in pred_tokens if token in prompt_tokens) / len(
    pred_tokens)


def mean(rows, key):
  if not rows:
    return math.nan
  return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def grouped_by_model_and_problem(scored):
  grouped = {}
  for row in scored:
    grouped.setdefault(row['model'], {}).setdefault(
      row['benchmark_id'], []).append(row)
  for by_problem in grouped.values():
    for rows in by_problem.values():
      rows.sort(key=lambda item: (
        item.get('sample_idx', 0),
        item.get('prompt_variant_idx', 0)))
  return grouped


def mean_permutation_stability(problem_rows):
  by_sample = {}
  for row in problem_rows:
    by_sample.setdefault(row.get('sample_idx', 0), []).append(row)

  scores = []
  for rows in by_sample.values():
    variant_rows = {}
    for row in rows:
      variant_rows[row.get('prompt_variant_idx', 0)] = row
    if len(variant_rows) < 2:
      continue
    values = list(variant_rows.values())
    for left_index in range(len(values)):
      for right_index in range(left_index + 1, len(values)):
        scores.append(emoji_metrics.bag_jaccard(
          values[left_index]['generated_emoji'],
          values[right_index]['generated_emoji']))
  return sum(scores) / len(scores) if scores else math.nan


def make_summary(scored):
  summaries = []
  for model, by_problem in grouped_by_model_and_problem(scored).items():
    rows = [row for problem_rows in by_problem.values() for row in problem_rows]
    first_rows = [
      problem_rows[0]
      for problem_rows in by_problem.values()
      if problem_rows]
    best_rows = [
      max(problem_rows, key=lambda item: item['benchmark_score'])
      for problem_rows in by_problem.values()
      if problem_rows]
    stability_rows = []
    for problem_rows in by_problem.values():
      stability = mean_permutation_stability(problem_rows)
      if not math.isnan(stability):
        stability_rows.append({'permutation_stability': stability})
    summary = {
      'model': model,
      'problems': len(by_problem),
      'samples': len(rows),
      'first_benchmark_score': mean(first_rows, 'benchmark_score'),
      'first_bag_f1': mean(first_rows, 'bag_f1'),
      'first_bag_jaccard': mean(first_rows, 'bag_jaccard'),
      'first_exact_bag_match': mean(first_rows, 'exact_bag_match'),
      'best7_benchmark_score': mean(best_rows, 'benchmark_score'),
      'best7_bag_f1': mean(best_rows, 'bag_f1'),
      'best7_bag_jaccard': mean(best_rows, 'bag_jaccard'),
      'first_copy_rate': mean(first_rows, 'copy_rate'),
      'mean_copy_rate': mean(rows, 'copy_rate'),
      'best7_copy_rate': mean(best_rows, 'copy_rate'),
      'pass_at_7_exact_bag': sum(
        1.0 for rows_for_problem in by_problem.values()
        if any(row['exact_bag_match'] >= 1.0 for row in rows_for_problem)
      ) / len(by_problem),
      'permutation_stability': mean(
        stability_rows, 'permutation_stability'),
      'permutation_stability_n': len(stability_rows),
      'permutation_stability_status': (
        'ok' if stability_rows else 'not_applicable'),
      'invalid_text_ratio': mean(rows, 'invalid_text_ratio'),
    }
    summaries.append(summary)
  summaries.sort(key=lambda row: row['best7_benchmark_score'], reverse=True)
  return summaries


def make_curves(scored, max_k):
  curves = []
  for model, by_problem in grouped_by_model_and_problem(scored).items():
    for k in range(1, max_k + 1):
      problem_count = len(by_problem)
      pass_exact_bag = 0.0
      pass_exact_sequence = 0.0
      best_score = 0.0
      best_f1 = 0.0
      best_jaccard = 0.0
      for rows in by_problem.values():
        prefix = rows[:k]
        if not prefix:
          continue
        pass_exact_bag += float(
          any(row['exact_bag_match'] >= 1.0 for row in prefix))
        pass_exact_sequence += float(
          any(row['exact_sequence_match'] >= 1.0 for row in prefix))
        best_score += max(row['benchmark_score'] for row in prefix)
        best_f1 += max(row['bag_f1'] for row in prefix)
        best_jaccard += max(row['bag_jaccard'] for row in prefix)
      curves.append({
        'model': model,
        'k': k,
        'pass_at_k_exact_bag': pass_exact_bag / problem_count,
        'pass_at_k_exact_sequence': pass_exact_sequence / problem_count,
        'pass_power_at_k_benchmark_score': best_score / problem_count,
        'best_bag_f1_at_k': best_f1 / problem_count,
        'best_bag_jaccard_at_k': best_jaccard / problem_count,
      })
  curves.sort(key=lambda row: (row['model'], row['k']))
  return curves


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


def write_report(path, summaries, curves, args):
  summary_columns = [
    'model',
    'problems',
    'samples',
    'first_benchmark_score',
    'best7_benchmark_score',
    'pass_at_7_exact_bag',
    'best7_bag_jaccard',
    'first_copy_rate',
    'mean_copy_rate',
    'best7_copy_rate',
    'permutation_stability',
    'permutation_stability_n',
    'permutation_stability_status',
    'invalid_text_ratio',
  ]
  curve_columns = [
    'model',
    'k',
    'pass_at_k_exact_bag',
    'pass_power_at_k_benchmark_score',
    'best_bag_jaccard_at_k',
  ]
  with open(path, 'w', encoding='utf-8') as f:
    f.write('# OpenRouter Emoji Eval\n\n')
    f.write(f'- Benchmark: `{args.benchmark}`\n')
    f.write(f'- Prompt mode: `{args.prompt_mode}`\n')
    f.write(f'- Prompt variants: `{args.prompt_variants}`\n')
    f.write(f'- Samples per problem: `{args.samples}`\n')
    f.write('- Thinking request: `reasoning_effort=none`, '
            '`include_reasoning=false`, `reasoning.exclude=true`\n\n')
    f.write('## Summary\n\n')
    f.write(markdown_table(summaries, summary_columns))
    f.write('\n\n## Pass / Power Curves\n\n')
    f.write(markdown_table(curves, curve_columns))
    f.write('\n')


def write_curve_svg(path, curves, y_key, title):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  width = 900
  height = 520
  left = 90
  right = 30
  top = 55
  bottom = 70
  plot_w = width - left - right
  plot_h = height - top - bottom
  models = sorted(set(row['model'] for row in curves))
  max_k = max((int(row['k']) for row in curves), default=1)
  colors = [
    '#2563eb',
    '#dc2626',
    '#16a34a',
    '#9333ea',
    '#ea580c',
    '#0891b2',
    '#4b5563',
  ]

  def x_for(k):
    if max_k <= 1:
      return left + plot_w / 2
    return left + (int(k) - 1) * plot_w / (max_k - 1)

  def y_for(value):
    value = max(0.0, min(1.0, float(value)))
    return top + (1.0 - value) * plot_h

  lines = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
    f'viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    f'<text x="{left}" y="30" font-family="sans-serif" font-size="22" '
    f'font-weight="700">{title}</text>',
    f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
    f'y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
    f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
    f'stroke="#111827" stroke-width="1"/>',
  ]
  for tick in range(0, 6):
    value = tick / 5
    y = y_for(value)
    lines.append(
      f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_w}" '
      f'y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
    lines.append(
      f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
      f'font-family="sans-serif" font-size="12">{value:.1f}</text>')
  for k in range(1, max_k + 1):
    x = x_for(k)
    lines.append(
      f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" '
      f'font-family="sans-serif" font-size="12">{k}</text>')
  lines.append(
    f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" '
    f'text-anchor="middle" font-family="sans-serif" font-size="14">k</text>')

  for index, model in enumerate(models):
    model_rows = sorted(
      [row for row in curves if row['model'] == model],
      key=lambda row: int(row['k']))
    color = colors[index % len(colors)]
    points = ' '.join(
      f"{x_for(row['k']):.1f},{y_for(row[y_key]):.1f}"
      for row in model_rows)
    lines.append(
      f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
      f'points="{points}"/>')
    legend_y = top + 20 + index * 22
    legend_x = left + plot_w - 190
    lines.append(
      f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" '
      f'y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
    lines.append(
      f'<text x="{legend_x + 30}" y="{legend_y + 4}" '
      f'font-family="sans-serif" font-size="13">{model}</text>')
  lines.append('</svg>')
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--benchmark',
                      default='data/emoji_reply/benchmark.jsonl')
  parser.add_argument('--out-dir',
                      default='outputs/emoji_reply/eval/openrouter')
  parser.add_argument('--model', action='append',
                      help='Model id or label=model_id. Use "default" for '
                           'the built-in frontier list.')
  parser.add_argument('--samples', type=int, default=7)
  parser.add_argument('--limit', type=int, default=None)
  parser.add_argument('--prompt-mode',
                      choices=[
                        'text_assisted',
                        'emoji_only',
                        'emoji_constrained',
                        'emoji_no_copy_reply'],
                      default='text_assisted')
  parser.add_argument('--prompt-variants', type=int, default=1,
                      help='Number of prompt permutations per benchmark row '
                           'when prompt-mode is emoji_only, '
                           'emoji_constrained, or emoji_no_copy_reply.')
  parser.add_argument('--temperature', type=float, default=0.7)
  parser.add_argument('--top-p', type=float, default=0.95)
  parser.add_argument('--max-tokens', type=int, default=24)
  parser.add_argument('--seed', type=int, default=20260620)
  parser.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
  parser.add_argument('--openrouter-url',
                      default='https://openrouter.ai/api/v1/chat/completions')
  parser.add_argument('--referer', default='https://github.com/ShujaRaina/EmojiModel')
  parser.add_argument('--title', default='EmojiModel OpenRouter Eval')
  parser.add_argument('--timeout', type=int, default=90)
  parser.add_argument('--retries', type=int, default=2)
  parser.add_argument('--retry-sleep', type=float, default=2.0)
  parser.add_argument('--sleep', type=float, default=0.0)
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--progress-interval', type=int, default=50)
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--score-only', action='store_true',
                      help='Skip API calls and score openrouter_raw_samples.jsonl.')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()

  benchmark_rows = read_jsonl(args.benchmark)
  if args.limit:
    benchmark_rows = benchmark_rows[:args.limit]
  models = parse_models(args.model)
  os.makedirs(args.out_dir, exist_ok=True)

  raw_path = os.path.join(args.out_dir, 'openrouter_raw_samples.jsonl')
  if args.score_only:
    sample_rows = read_jsonl(raw_path)
  else:
    sample_rows = generate_predictions(benchmark_rows, models, args)

  scored = score_samples(benchmark_rows, sample_rows)
  summaries = make_summary(scored)
  curves = make_curves(scored, args.samples)

  write_jsonl(os.path.join(args.out_dir, 'openrouter_scored_samples.jsonl'), scored)
  write_csv(os.path.join(args.out_dir, 'openrouter_summary.csv'), summaries)
  write_csv(os.path.join(args.out_dir, 'openrouter_pass_curves.csv'), curves)
  write_report(os.path.join(args.out_dir, 'OPENROUTER_RESULTS.md'),
               summaries, curves, args)
  write_curve_svg(
    os.path.join(args.out_dir, 'pass_at_k_exact_bag.svg'),
    curves,
    'pass_at_k_exact_bag',
    'Pass@k Exact Bag Match')
  write_curve_svg(
    os.path.join(args.out_dir, 'pass_power_at_k_benchmark_score.svg'),
    curves,
    'pass_power_at_k_benchmark_score',
    'Best-of-k Benchmark Score')

  print(markdown_table(summaries, [
    'model',
    'problems',
    'samples',
    'first_benchmark_score',
    'best7_benchmark_score',
    'pass_at_7_exact_bag',
    'best7_bag_jaccard',
    'first_copy_rate',
    'mean_copy_rate',
    'best7_copy_rate',
    'permutation_stability',
    'permutation_stability_n',
    'permutation_stability_status',
    'invalid_text_ratio',
  ]))
  print()
  print(f"Wrote results to {args.out_dir}")


if __name__ == '__main__':
  main()
