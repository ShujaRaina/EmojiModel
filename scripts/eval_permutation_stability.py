"""Evaluate emoji-order stability on curated permutation groups.

The challenge file contains groups where prompt permutations should lead to
semantically equivalent response emoji bags. This script samples each prompt
from one or more checkpoints and reports pairwise bag-Jaccard stability within
each group.

Example:
  python scripts/eval_permutation_stability.py \
    --checkpoint mdlm=outputs/emoji_two_phase_h100/phase2_reply/checkpoints/last.ckpt \
    --data-cache /tmp/emoji_mdlm_two_phase \
    --vocab-cache /tmp/emoji_mdlm_two_phase/atomic_emoji_vocab.json \
    --steps 64
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

import hydra
import omegaconf
import torch
import torch.nn.functional as F

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd, replace=True)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', lambda: max(torch.cuda.device_count(), 1), replace=True)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval, replace=True)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y, replace=True)

_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
  kwargs.setdefault('weights_only', False)
  return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

import dataloader
import diffusion
from scripts import emoji_metrics


def parse_checkpoint(value):
  if '=' in value:
    label, path = value.split('=', 1)
    return label, path
  path = value
  label = os.path.splitext(os.path.basename(path))[0]
  return label, path


def read_groups(path):
  groups = collections.defaultdict(list)
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      groups[row.get('canonical_id', row.get('topic', 'ungrouped'))].append(row)
  return groups


def _emoji_text(value):
  return ''.join(dataloader.extract_emoji_graphemes(value or ''))


def _reply_row_split_key(row, strategy):
  return dataloader._emoji_reply_split_key(row, strategy)


def _partition_reply_rows(rows, strategy, validation_size, split_seed,
                          partition):
  if strategy == 'random' or partition == 'all':
    return rows
  keys = [_reply_row_split_key(row, strategy) for _, row in rows]
  unique_keys = sorted(set(keys))
  if len(unique_keys) <= 1:
    return rows

  def key_sort_value(key):
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


def read_reply_data_groups(path, num_groups, permutations_per_group, seed,
                           split_strategy='random',
                           validation_size=0.05,
                           split_seed=42,
                           split_partition='all'):
  """Build prompt permutation groups from the repo reply dataset only."""
  rng = random.Random(seed)
  groups = collections.OrderedDict()
  rows = []
  with open(path, 'r', encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      if not line.strip():
        continue
      row = json.loads(line)
      rows.append((line_no, row))
  rows = _partition_reply_rows(
    rows,
    split_strategy,
    validation_size,
    split_seed,
    split_partition)
  for line_no, row in rows:
    if len(groups) >= num_groups:
      break
    prompt_tokens = dataloader.extract_emoji_graphemes(
      row.get('input') or row.get('prompt_emoji') or row.get('emoji') or '')
    response = _emoji_text(
      row.get('output') or row.get('response_emoji') or row.get('emoji') or '')
    if len(prompt_tokens) < 2 or not response:
      continue

    variants = []
    seen = set()
    original = ''.join(prompt_tokens)
    variants.append(original)
    seen.add(original)
    attempts = 0
    while len(variants) < permutations_per_group and attempts < 64:
      attempts += 1
      shuffled = list(prompt_tokens)
      rng.shuffle(shuffled)
      candidate = ''.join(shuffled)
      if candidate not in seen:
        variants.append(candidate)
        seen.add(candidate)

    if len(variants) < 2:
      continue

    group_id = f"reply_{line_no:05d}_{row.get('topic', 'untagged')}"
    groups[group_id] = [
      {
        'canonical_id': group_id,
        'topic': row.get('topic', ''),
        'instruction': row.get('instruction', ''),
        'prompt_emoji': prompt,
        'response_emoji': response,
      }
      for prompt in variants
    ]
  return groups


def build_config(args, checkpoint):
  overrides = [
    'mode=sample_eval',
    'data=emoji_reply',
    f'model={args.model}',
    f'model.length={args.length}',
    f'parameterization={args.parameterization}',
    f'backbone={args.backbone}',
    f'trainer.accelerator={args.device}',
    'trainer.devices=1',
    f'sampling.steps={args.steps}',
    f'eval.checkpoint_path={checkpoint}',
    f'data.cache_dir={args.data_cache}',
    f'data.data_file={os.path.abspath(args.reply_data_file)}',
    'data.emoji_vocab_sources=[text2emoji,common]',
    f'data.emoji_vocab_extra_files=[{os.path.abspath(args.reply_data_file)}]',
    f'data.emoji_challenge_set_path={os.path.abspath(args.challenge_file)}',
    'data.emoji_include_challenge_in_train=false',
  ]
  if args.vocab_cache:
    overrides.append(f'data.emoji_vocab_cache={args.vocab_cache}')
  return hydra.compose(config_name='config', overrides=overrides)


def prefix_for_prompt(tokenizer, prompt, length):
  prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
  prompt_ids = prompt_ids[:length - 2]
  return [tokenizer.bos_token_id] + prompt_ids + [tokenizer.sep_token_id]


def extract_response_with_stop_stats(tokenizer, token_ids, prefix_len):
  tail = token_ids[prefix_len:]
  eos_rel = None
  pad_rel = None
  for offset, token_id in enumerate(tail):
    if eos_rel is None and token_id == tokenizer.eos_token_id:
      eos_rel = offset
    if pad_rel is None and token_id == tokenizer.pad_token_id:
      pad_rel = offset
    if eos_rel is not None and pad_rel is not None:
      break

  out = []
  for token_id in tail:
    if token_id in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
      break
    out.append(token_id)
  return {
    'generated_emoji': tokenizer.decode(out).strip(),
    'eos_rel': eos_rel,
    'pad_rel': pad_rel,
    'decoded_len': len(out),
  }


def extract_response(tokenizer, token_ids, prefix_len):
  return extract_response_with_stop_stats(
    tokenizer, token_ids, prefix_len)['generated_emoji']


@torch.no_grad()
def ar_cond_sample(model, tokenizer, prefix_ids, length):
  samples = []
  for prefix in prefix_ids:
    prefix = prefix[:length]
    x = torch.full(
      (1, length),
      tokenizer.pad_token_id,
      dtype=torch.long,
      device=model.device)
    x[0, :len(prefix)] = torch.tensor(
      prefix, dtype=torch.long, device=model.device)
    cursor = len(prefix)
    while cursor < length:
      logits = model.forward(x[:, :cursor], None)[:, -1]
      next_id = logits.argmax(dim=-1)
      x[0, cursor] = next_id
      cursor += 1
      if int(next_id.item()) == tokenizer.eos_token_id:
        break
    samples.append(x[0].detach().cpu().tolist())
  return samples


def sample_model(args, label, checkpoint, groups):
  with hydra.initialize(version_base=None, config_path='../configs'):
    config = build_config(args, checkpoint)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    checkpoint, tokenizer=tokenizer, config=config)
  model.to(args.device)
  model.eval()

  rows = list(itertools.chain.from_iterable(groups.values()))
  prompts = [row['prompt_emoji'] for row in rows]
  prefix_ids = [
    prefix_for_prompt(tokenizer, prompt, args.length)
    for prompt in prompts]
  prefix_lens = [len(prefix) for prefix in prefix_ids]
  if args.parameterization == 'ar' or args.backbone == 'ar':
    sample_ids = ar_cond_sample(model, tokenizer, prefix_ids, args.length)
  else:
    sample_ids = model.restore_model_and_cond_sample(
      prefix_ids,
      num_steps=args.steps,
      max_response_tokens=args.max_response_tokens).cpu().tolist()

  outputs_by_group = collections.defaultdict(list)
  records = []
  for row, ids, prefix_len in zip(rows, sample_ids, prefix_lens):
    stop_stats = extract_response_with_stop_stats(tokenizer, ids, prefix_len)
    record = {
      'model': label,
      'canonical_id': row.get('canonical_id', row.get('topic', 'ungrouped')),
      'topic': row.get('topic', ''),
      'instruction': row.get('instruction', ''),
      'prompt_emoji': row['prompt_emoji'],
      'target_emoji': row.get('response_emoji', row.get('emoji', '')),
      **stop_stats,
    }
    target_metrics = emoji_metrics.score_pair(
      record['generated_emoji'], record['target_emoji'])
    for key, value in target_metrics.items():
      if key != 'matched_reference':
        record[f'target_{key}'] = value
    outputs_by_group[record['canonical_id']].append(record)
    records.append(record)
  return outputs_by_group, records


class EmojiSemanticScorer:
  """Offline semantic scorer from Text2Emoji/BGE table embeddings."""

  def __init__(self, table_path):
    payload = torch.load(table_path, map_location='cpu')
    rows = payload['rows']
    embeddings = F.normalize(payload['text_embedding'].float(), dim=-1)
    token_vectors = collections.defaultdict(list)
    for row, embedding in zip(rows, embeddings):
      for token in set(dataloader.extract_emoji_graphemes(row.get('emoji', ''))):
        token_vectors[token].append(embedding)
    self.token_embedding = {}
    for token, vectors in token_vectors.items():
      self.token_embedding[token] = F.normalize(
        torch.stack(vectors, dim=0).mean(dim=0), dim=0)
    if self.token_embedding:
      center = torch.stack(list(self.token_embedding.values()), dim=0).mean(
        dim=0)
      self.token_embedding = {
        token: F.normalize(embedding - center, dim=0)
        for token, embedding in self.token_embedding.items()
      }

  def embed(self, text):
    vectors = [
      self.token_embedding[token]
      for token in dataloader.extract_emoji_graphemes(text)
      if token in self.token_embedding]
    if not vectors:
      return None
    return F.normalize(torch.stack(vectors, dim=0).mean(dim=0), dim=0)

  def similarity(self, left, right):
    left_embedding = self.embed(left)
    right_embedding = self.embed(right)
    if left_embedding is None and right_embedding is None:
      return 1.0
    if left_embedding is None or right_embedding is None:
      return 0.0
    return float(torch.dot(left_embedding, right_embedding).item())


def pairwise_stability(records, scorer=None):
  if len(records) < 2:
    return 1.0
  scores = []
  for left, right in itertools.combinations(records, 2):
    if scorer is None:
      scores.append(emoji_metrics.bag_jaccard(
        left['generated_emoji'], right['generated_emoji']))
    else:
      scores.append(scorer.similarity(
        left['generated_emoji'], right['generated_emoji']))
  return sum(scores) / len(scores)


def target_alignment(records, scorer=None):
  if not records:
    return 0.0
  if scorer is None:
    return sum(
      emoji_metrics.bag_jaccard(r['generated_emoji'], r['target_emoji'])
      for r in records) / len(records)
  return sum(
    scorer.similarity(r['generated_emoji'], r['target_emoji'])
    for r in records) / len(records)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', action='append', required=True,
                      help='Checkpoint path, optionally label=path.')
  parser.add_argument('--challenge-file',
                      default='data/emoji_reply/curated_permutation.jsonl')
  parser.add_argument('--source', choices=['reply_data', 'challenge'],
                      default='reply_data')
  parser.add_argument('--reply-data-file',
                      default='data/emoji_reply/emoji_reply.jsonl')
  parser.add_argument('--num-groups', type=int, default=24)
  parser.add_argument('--permutations-per-group', type=int, default=3)
  parser.add_argument('--split-strategy',
                      choices=[
                        'random',
                        'prompt_bag',
                        'response_bag',
                        'prompt_response_bag'],
                      default='random')
  parser.add_argument('--split-partition',
                      choices=['all', 'train', 'validation'],
                      default='all')
  parser.add_argument('--split-validation-size', type=float, default=0.05)
  parser.add_argument('--split-seed', type=int, default=42)
  parser.add_argument('--data-cache', default='/tmp/emoji_mdlm_two_phase')
  parser.add_argument('--vocab-cache', default=None)
  parser.add_argument('--semantic-table', default=None,
                      help='Optional Text2Emoji/BGE semantic_table.pt for '
                           'offline semantic cosine eval.')
  parser.add_argument('--jsonl-out', default=None)
  parser.add_argument('--steps', type=int, default=64)
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--model', default='small')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  parser.add_argument('--max-response-tokens', type=int, default=None,
                      help='Optionally clamp EOS after this many response emoji.')
  parser.add_argument('--backbone', default='dit')
  parser.add_argument('--parameterization', default='subs')
  parser.add_argument('--seed', type=int, default=1)
  args = parser.parse_args()

  random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  args.challenge_file = os.path.abspath(args.challenge_file)
  args.reply_data_file = os.path.abspath(args.reply_data_file)
  if args.source == 'challenge':
    groups = read_groups(args.challenge_file)
  else:
    groups = read_reply_data_groups(
      args.reply_data_file,
      args.num_groups,
      args.permutations_per_group,
      args.seed,
      split_strategy=args.split_strategy,
      validation_size=args.split_validation_size,
      split_seed=args.split_seed,
      split_partition=args.split_partition)
  if not groups:
    raise ValueError(f'No permutation groups found for source={args.source}')
  semantic_scorer = (
    EmojiSemanticScorer(args.semantic_table)
    if args.semantic_table else None)
  all_records = []
  summaries = []
  for checkpoint_arg in args.checkpoint:
    label, checkpoint = parse_checkpoint(checkpoint_arg)
    outputs_by_group, records = sample_model(args, label, checkpoint, groups)
    if semantic_scorer is not None:
      for record in records:
        record['semantic_target_cosine'] = semantic_scorer.similarity(
          record['generated_emoji'], record['target_emoji'])
    all_records.extend(records)
    group_scores = {
      group_id: pairwise_stability(group_records)
      for group_id, group_records in outputs_by_group.items()
    }
    target_jaccard_scores = {
      group_id: target_alignment(group_records)
      for group_id, group_records in outputs_by_group.items()
    }
    target_benchmark_scores = {
      group_id: (
        sum(r['target_benchmark_score'] for r in group_records)
        / len(group_records))
      for group_id, group_records in outputs_by_group.items()
    }
    target_f1_scores = {
      group_id: (
        sum(r['target_bag_f1'] for r in group_records)
        / len(group_records))
      for group_id, group_records in outputs_by_group.items()
    }
    semantic_group_scores = {}
    semantic_target_scores = {}
    if semantic_scorer is not None:
      semantic_group_scores = {
        group_id: pairwise_stability(group_records, semantic_scorer)
        for group_id, group_records in outputs_by_group.items()
      }
      semantic_target_scores = {
        group_id: target_alignment(group_records, semantic_scorer)
        for group_id, group_records in outputs_by_group.items()
      }
    summaries.append({
      'model': label,
      'mean_permutation_stability': (
        sum(group_scores.values()) / len(group_scores)),
      'mean_target_benchmark_score': (
        sum(target_benchmark_scores.values()) / len(target_benchmark_scores)),
      'mean_target_bag_f1': (
        sum(target_f1_scores.values()) / len(target_f1_scores)),
      'mean_target_bag_jaccard': (
        sum(target_jaccard_scores.values()) / len(target_jaccard_scores)),
      'mean_semantic_permutation_stability': (
        sum(semantic_group_scores.values()) / len(semantic_group_scores)
        if semantic_group_scores else None),
      'mean_semantic_target_cosine': (
        sum(semantic_target_scores.values()) / len(semantic_target_scores)
        if semantic_target_scores else None),
      'groups': group_scores,
    })

  if args.jsonl_out:
    os.makedirs(os.path.dirname(args.jsonl_out) or '.', exist_ok=True)
    with open(args.jsonl_out, 'w', encoding='utf-8') as f:
      for record in all_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

  header = [
    'model',
    'perm_stability',
    'target_score',
    'target_bag_f1',
    'target_bag_jaccard']
  if semantic_scorer is not None:
    header.extend(['semantic_stability', 'semantic_target_cosine'])
  print('\t'.join(header))
  for summary in summaries:
    values = [
      summary['model'],
      f"{summary['mean_permutation_stability']:.4f}",
      f"{summary['mean_target_benchmark_score']:.4f}",
      f"{summary['mean_target_bag_f1']:.4f}",
      f"{summary['mean_target_bag_jaccard']:.4f}",
    ]
    if semantic_scorer is not None:
      values.extend([
        f"{summary['mean_semantic_permutation_stability']:.4f}",
        f"{summary['mean_semantic_target_cosine']:.4f}",
      ])
    print('\t'.join(values))
  print('\nSample generations:')
  for record in all_records[:12]:
    print(
      f"{record['model']} {record['canonical_id']} "
      f"{record['prompt_emoji']} -> {record['generated_emoji']} "
      f"(target {record['target_emoji']}, "
      f"eos_rel={record['eos_rel']}, "
      f"pad_rel={record['pad_rel']}, "
      f"decoded_len={record['decoded_len']})")


if __name__ == '__main__':
  main()
