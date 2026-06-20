"""Conditional emoji generation from a trained MDLM checkpoint.

Given one or more text prompts, this script clamps the
`<eot> text <eot>` prefix and lets the masked diffusion model infill the
emoji span (everything up to the next `<eot>`).

Example:
  python scripts/sample_emoji.py \
    --checkpoint outputs/emoji_run/checkpoints/last.ckpt \
    --steps 128 \
    --prompt "I love sunny beaches and ice cream" \
    --prompt "Feeling sad and lonely today"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch

# Lightning checkpoints store the omegaconf config; allow the full (trusted,
# locally-produced) unpickle since torch>=2.6 defaults to weights_only=True.
_orig_torch_load = torch.load


def _torch_load(*a, **k):
  k.setdefault('weights_only', False)
  return _orig_torch_load(*a, **k)


torch.load = _torch_load

import dataloader
import diffusion


def build_config(overrides):
  with hydra.initialize(version_base=None, config_path='../configs'):
    return hydra.compose(config_name='config', overrides=overrides)


def extract_emoji(token_ids, prefix_len, eot_id):
  """Returns the token span between the 2nd and 3rd `<eot>`."""
  gen = token_ids[prefix_len:]
  out = []
  for tok in gen:
    if tok == eot_id:
      break
    out.append(tok)
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--prompt', action='append', default=[],
                      help='Text prompt (repeatable).')
  parser.add_argument('--steps', type=int, default=128)
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--model', default='tiny-emoji')
  args = parser.parse_args()

  prompts = args.prompt or [
    'I love sunny beaches and ice cream',
    'Feeling sad and lonely today',
    'Time to study hard for my exams',
    'Happy birthday to my best friend',
  ]

  torch.manual_seed(args.seed)
  config = build_config([
    'mode=sample_eval',
    'data=text2emoji',
    f'model={args.model}',
    f'model.length={args.length}',
    'parameterization=subs',
    'backbone=dit',
    'trainer.accelerator=cpu',
    'trainer.devices=1',
    f'sampling.steps={args.steps}',
    f'eval.checkpoint_path={args.checkpoint}',
  ])

  tokenizer = dataloader.get_tokenizer(config)
  eot_id = tokenizer.eos_token_id
  model = diffusion.Diffusion.load_from_checkpoint(
    args.checkpoint, tokenizer=tokenizer, config=config)
  model.eval()

  prefix_ids, prefix_lens = [], []
  for text in prompts:
    text_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    text_ids = text_ids[:args.length - 3]
    prefix = [eot_id] + text_ids + [eot_id]
    prefix_ids.append(prefix)
    prefix_lens.append(len(prefix))

  samples = model.restore_model_and_cond_sample(
    prefix_ids, num_steps=args.steps)
  samples = samples.cpu().tolist()

  print('\n=== Emoji generations ===')
  for text, ids, plen in zip(prompts, samples, prefix_lens):
    emoji_ids = extract_emoji(ids, plen, eot_id)
    emoji = tokenizer.decode(emoji_ids).strip()
    print(f'{text!r:60s} -> {emoji}')


if __name__ == '__main__':
  main()
