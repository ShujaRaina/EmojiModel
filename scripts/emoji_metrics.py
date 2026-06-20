"""Shared metrics for emoji reply benchmark evaluation."""
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

import dataloader


def emoji_tokens(text):
  return dataloader.extract_emoji_graphemes(text or '')


def emoji_text(text):
  return ''.join(emoji_tokens(text))


def emoji_bag(text):
  return collections.Counter(emoji_tokens(text))


def _bag_overlap(pred_bag, target_bag):
  keys = set(pred_bag) | set(target_bag)
  return sum(min(pred_bag[k], target_bag[k]) for k in keys)


def bag_jaccard(prediction, target):
  """Multiset Jaccard: overlap divided by multiset union."""
  pred_bag = emoji_bag(prediction)
  target_bag = emoji_bag(target)
  if not pred_bag and not target_bag:
    return 1.0
  keys = set(pred_bag) | set(target_bag)
  union = sum(max(pred_bag[k], target_bag[k]) for k in keys)
  return _bag_overlap(pred_bag, target_bag) / union if union else 0.0


def set_jaccard(prediction, target):
  """Unique-token Jaccard, useful when order and repeated emojis should matter less."""
  pred_set = set(emoji_tokens(prediction))
  target_set = set(emoji_tokens(target))
  if not pred_set and not target_set:
    return 1.0
  union = pred_set | target_set
  return len(pred_set & target_set) / len(union) if union else 0.0


def bag_precision_recall_f1(prediction, target):
  pred_bag = emoji_bag(prediction)
  target_bag = emoji_bag(target)
  overlap = _bag_overlap(pred_bag, target_bag)
  pred_total = sum(pred_bag.values())
  target_total = sum(target_bag.values())
  precision = overlap / pred_total if pred_total else (1.0 if not target_total else 0.0)
  recall = overlap / target_total if target_total else (1.0 if not pred_total else 0.0)
  f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0)
  return precision, recall, f1


def length_score(prediction, target):
  pred_len = len(emoji_tokens(prediction))
  target_len = len(emoji_tokens(target))
  if pred_len == 0 and target_len == 0:
    return 1.0
  if pred_len == 0 or target_len == 0:
    return 0.0
  return min(pred_len, target_len) / max(pred_len, target_len)


def exact_bag_match(prediction, target):
  return float(emoji_bag(prediction) == emoji_bag(target))


def exact_sequence_match(prediction, target):
  return float(emoji_tokens(prediction) == emoji_tokens(target))


def invalid_text_ratio(raw_prediction):
  raw_prediction = raw_prediction or ''
  if not raw_prediction:
    return 0.0
  emoji_chars = ''.join(emoji_tokens(raw_prediction))
  return max(0, len(raw_prediction.strip()) - len(emoji_chars)) / max(
    1, len(raw_prediction.strip()))


def score_pair(prediction, target):
  precision, recall, f1 = bag_precision_recall_f1(prediction, target)
  jaccard = bag_jaccard(prediction, target)
  unique_jaccard = set_jaccard(prediction, target)
  len_score = length_score(prediction, target)
  # Primary benchmark score: smoother than pure Jaccard but still grounded in
  # target overlap, with small penalties for poor length control.
  benchmark_score = (
    0.50 * f1
    + 0.25 * jaccard
    + 0.15 * unique_jaccard
    + 0.10 * len_score)
  return {
    'benchmark_score': benchmark_score,
    'bag_f1': f1,
    'bag_precision': precision,
    'bag_recall': recall,
    'bag_jaccard': jaccard,
    'set_jaccard': unique_jaccard,
    'length_score': len_score,
    'exact_bag_match': exact_bag_match(prediction, target),
    'exact_sequence_match': exact_sequence_match(prediction, target),
    'prediction_len': len(emoji_tokens(prediction)),
    'target_len': len(emoji_tokens(target)),
  }


def best_reference_score(prediction, references):
  references = [ref for ref in references if emoji_tokens(ref)]
  if not references:
    references = ['']
  scored = [
    (score_pair(prediction, reference), reference)
    for reference in references]
  scored.sort(
    key=lambda item: (
      item[0]['benchmark_score'],
      item[0]['bag_f1'],
      item[0]['bag_jaccard']),
    reverse=True)
  metrics, reference = scored[0]
  metrics = dict(metrics)
  metrics['matched_reference'] = reference
  return metrics


def mean_dict(rows, keys):
  if not rows:
    return {key: math.nan for key in keys}
  return {
    key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    for key in keys}
