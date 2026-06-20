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


def read_reply_data_groups(path, num_groups, permutations_per_group, seed):
  """Build prompt permutation groups from the repo reply dataset only."""
  rng = random.Random(seed)
  groups = collections.OrderedDict()
  with open(path, 'r', encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      if len(groups) >= num_groups:
        break
      if not line.strip():
        continue
      row = json.loads(line)
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


def extract_response(tokenizer, token_ids, prefix_len):
  out = []
  for token_id in token_ids[prefix_len:]:
    if token_id in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
      break
    out.append(token_id)
  return tokenizer.decode(out).strip()


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
    generated = extract_response(tokenizer, ids, prefix_len)
    record = {
      'model': label,
      'canonical_id': row.get('canonical_id', row.get('topic', 'ungrouped')),
      'topic': row.get('topic', ''),
      'instruction': row.get('instruction', ''),
      'prompt_emoji': row['prompt_emoji'],
      'target_emoji': row.get('response_emoji', row.get('emoji', '')),
      'generated_emoji': generated,
    }
    outputs_by_group[record['canonical_id']].append(record)
    records.append(record)
  return outputs_by_group, records


def emoji_bag(text):
  return collections.Counter(dataloader.extract_emoji_graphemes(text))


def bag_jaccard(a, b):
  a = emoji_bag(a)
  b = emoji_bag(b)
  if not a and not b:
    return 1.0
  keys = set(a) | set(b)
  inter = sum(min(a[k], b[k]) for k in keys)
  union = sum(max(a[k], b[k]) for k in keys)
  return inter / union if union else 0.0


def pairwise_stability(records):
  if len(records) < 2:
    return 1.0
  scores = []
  for left, right in itertools.combinations(records, 2):
    scores.append(bag_jaccard(
      left['generated_emoji'], right['generated_emoji']))
  return sum(scores) / len(scores)


def target_alignment(records):
  if not records:
    return 0.0
  return sum(
    bag_jaccard(r['generated_emoji'], r['target_emoji'])
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
  parser.add_argument('--data-cache', default='/tmp/emoji_mdlm_two_phase')
  parser.add_argument('--vocab-cache', default=None)
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

  args.challenge_file = os.path.abspath(args.challenge_file)
  args.reply_data_file = os.path.abspath(args.reply_data_file)
  if args.source == 'challenge':
    groups = read_groups(args.challenge_file)
  else:
    groups = read_reply_data_groups(
      args.reply_data_file,
      args.num_groups,
      args.permutations_per_group,
      args.seed)
  if not groups:
    raise ValueError(f'No permutation groups found for source={args.source}')
  all_records = []
  summaries = []
  for checkpoint_arg in args.checkpoint:
    label, checkpoint = parse_checkpoint(checkpoint_arg)
    outputs_by_group, records = sample_model(args, label, checkpoint, groups)
    all_records.extend(records)
    group_scores = {
      group_id: pairwise_stability(group_records)
      for group_id, group_records in outputs_by_group.items()
    }
    target_scores = {
      group_id: target_alignment(group_records)
      for group_id, group_records in outputs_by_group.items()
    }
    summaries.append({
      'model': label,
      'mean_permutation_stability': (
        sum(group_scores.values()) / len(group_scores)),
      'mean_target_bag_jaccard': (
        sum(target_scores.values()) / len(target_scores)),
      'groups': group_scores,
    })

  if args.jsonl_out:
    os.makedirs(os.path.dirname(args.jsonl_out) or '.', exist_ok=True)
    with open(args.jsonl_out, 'w', encoding='utf-8') as f:
      for record in all_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

  print('model\tperm_stability\ttarget_bag_jaccard')
  for summary in summaries:
    print(
      f"{summary['model']}\t"
      f"{summary['mean_permutation_stability']:.4f}\t"
      f"{summary['mean_target_bag_jaccard']:.4f}")
  print('\nSample generations:')
  for record in all_records[:12]:
    print(
      f"{record['model']} {record['canonical_id']} "
      f"{record['prompt_emoji']} -> {record['generated_emoji']} "
      f"(target {record['target_emoji']})")


if __name__ == '__main__':
  main()
