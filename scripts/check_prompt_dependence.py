"""Does a dataset's reply actually depend on its prompt?

This is the test that would have caught the confound in
`scripts/build_emoji_reply_dataset.py`, where a prompt and its reply are two
independent draws from the same topic pool. In that dataset a reply is no more
related to its own prompt than to any other prompt, so a model trained on it
cannot be learning a prompt -> reply mapping, and any order-invariance it shows
is a property of the generator rather than a finding about emoji.

Method
------
Score how similar each reply is to its true prompt, then re-score after
shuffling replies among prompts. Two shuffles are reported:

  global   replies shuffled across the whole dataset
  within   replies shuffled only among prompts sharing a topic

`within` is the one that matters. A dataset can look prompt-dependent under a
global shuffle purely because topics differ; only beating a within-topic
shuffle shows a reply responds to *that* prompt rather than to its subject.

A dataset passes if `true` is clearly above `within`. `lift` is the gap in
standard deviations of the shuffled distribution.

Usage
-----
    python scripts/check_prompt_dependence.py data/emoji_reply/emoji_reply.jsonl
    python scripts/check_prompt_dependence.py a.jsonl b.jsonl --trials 20
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emoji_tokenization


def load(path):
  rows = []
  with open(path, encoding='utf-8') as handle:
    for line in handle:
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except ValueError:
        continue
      prompt = ''.join(emoji_tokenization.extract_emoji_graphemes(
        row.get('input') or ''))
      response = ''.join(emoji_tokenization.extract_emoji_graphemes(
        row.get('output') or ''))
      if prompt and response:
        rows.append((prompt, response, row.get('topic') or ''))
  return rows


def mean_similarity(pairs):
  if not pairs:
    return 0.0
  return sum(emoji_tokenization.bag_jaccard(p, r)
             for p, r in pairs) / len(pairs)


def shuffled_within(rows, rng):
  """Permute replies only among rows sharing a topic."""
  by_topic = collections.defaultdict(list)
  for index, (_, _, topic) in enumerate(rows):
    by_topic[topic].append(index)
  responses = [r for _, r, _ in rows]
  out = list(responses)
  for indices in by_topic.values():
    pool = [responses[i] for i in indices]
    rng.shuffle(pool)
    for slot, index in enumerate(indices):
      out[index] = pool[slot]
  return [(rows[i][0], out[i]) for i in range(len(rows))]


def shuffled_global(rows, rng):
  responses = [r for _, r, _ in rows]
  rng.shuffle(responses)
  return [(rows[i][0], responses[i]) for i in range(len(rows))]


def analyse(path, trials, seed):
  rows = load(path)
  if len(rows) < 20:
    print(f'{path}: only {len(rows)} usable rows — too few to judge')
    return None

  rng = random.Random(seed)
  true_score = mean_similarity([(p, r) for p, r, _ in rows])
  within = [mean_similarity(shuffled_within(rows, rng)) for _ in range(trials)]
  glob = [mean_similarity(shuffled_global(rows, rng)) for _ in range(trials)]

  within_mean = statistics.mean(within)
  within_sd = statistics.pstdev(within) or 1e-12
  lift = (true_score - within_mean) / within_sd

  topics = len({t for _, _, t in rows})
  print(f'\n{path}')
  print(f'  rows {len(rows)}   topics {topics}   trials {trials}')
  print(f'  true prompt-reply similarity   {true_score:.4f}')
  print(f'  shuffled globally              {statistics.mean(glob):.4f}')
  print(f'  shuffled within topic          {within_mean:.4f}  '
        f'(sd {within_sd:.4f})')
  print(f'  lift over within-topic shuffle {lift:+.1f} sd')

  if lift < 2:
    print('  VERDICT: FAIL — a reply is no closer to its own prompt than to a')
    print('           random reply from the same topic. This data carries no')
    print('           prompt->reply signal; only topic identity.')
  elif lift < 6:
    print('  VERDICT: WEAK — some prompt-specific signal, but most of the')
    print('           similarity is explained by topic.')
  else:
    print('  VERDICT: PASS — replies are measurably specific to their prompts.')
  return lift


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('paths', nargs='+', help='JSONL datasets to check')
  parser.add_argument('--trials', type=int, default=10)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  for path in args.paths:
    if not os.path.exists(path):
      print(f'{path}: not found')
      continue
    analyse(path, args.trials, args.seed)


if __name__ == '__main__':
  main()
