"""Canonical emoji grapheme segmentation and multiset overlap primitives.

This module is the single source of truth for how the project turns a string
into emoji tokens and how it scores two emoji strings against each other.

It deliberately depends on nothing beyond `regex` so that the frontier-model
evaluations under `eval/` can import it without pulling in torch,
transformers, or datasets. `dataloader.py` re-exports the segmentation
functions, so existing `dataloader.extract_emoji_graphemes` callers are
unaffected.

Keeping one implementation matters for correctness, not just tidiness: the
headline MDLM-vs-frontier comparisons score both sides with these functions,
and that comparison is only meaningful if both sides tokenize identically.
"""
import collections

try:
  import regex
except ImportError:  # pragma: no cover - only hit in incomplete envs.
  regex = None


def _require_regex():
  if regex is None:
    raise ImportError(
      'Atomic emoji tokenization requires the `regex` package. '
      'Install project requirements or run `pip install regex`.')


def split_graphemes(text):
  _require_regex()
  return regex.findall(r'\X', text or '')


def is_emoji_grapheme(grapheme):
  _require_regex()
  if not grapheme or grapheme.isspace():
    return False
  return bool(
    regex.search(r'\p{Extended_Pictographic}', grapheme)
    or regex.search(r'\p{Regional_Indicator}', grapheme)
    or '⃣' in grapheme
    or any('\U000E0020' <= ch <= '\U000E007F' for ch in grapheme))


def extract_emoji_graphemes(text):
  return [g for g in split_graphemes(text) if is_emoji_grapheme(g)]


def emoji_bag(text):
  return collections.Counter(extract_emoji_graphemes(text))


def bag_overlap(pred_bag, target_bag):
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
  return bag_overlap(pred_bag, target_bag) / union if union else 0.0


def best_jaccard(prediction, references):
  return max(
    (bag_jaccard(prediction, reference) for reference in references),
    default=0.0)
