"""Conditional emoji generation from a trained MDLM checkpoint.

Given one or more emoji prompts, this script clamps the
`[BOS] prompt_emoji [SEP]` prefix and lets the masked diffusion model infill
the response span up to `[EOS]`.

Example:
  python scripts/sample_emoji.py \
    --checkpoint outputs/emoji_run/checkpoints/last.ckpt \
    --steps 128 \
    --prompt "☀️🏖️🍦" \
    --prompt "😢💔"
"""
import argparse
import os
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


def extract_emoji(token_ids, prefix_len, eos_id, pad_id):
  """Returns generated response tokens after the clamped prefix."""
  gen = token_ids[prefix_len:]
  out = []
  for tok in gen:
    if tok == eos_id or tok == pad_id:
      break
    out.append(tok)
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--prompt', action='append', default=[],
                      help='Emoji prompt (repeatable).')
  parser.add_argument('--steps', type=int, default=128)
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--model', default='tiny-emoji')
  parser.add_argument('--data-cache', default='/tmp/emoji_mdlm_data')
  parser.add_argument('--vocab-cache', default=None)
  parser.add_argument('--data-file', default='data/emoji_reply/emoji_reply.jsonl')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  parser.add_argument('--max-response-tokens', type=int, default=3,
                      help='Clamp EOS after this many generated emoji.')
  args = parser.parse_args()

  prompts = args.prompt or [
    '☀️🏖️🍦',
    '😢💔',
    '📚💪',
    '🎂🎉',
  ]

  torch.manual_seed(args.seed)
  config = build_config([
    'mode=sample_eval',
    'data=emoji_reply',
    f'model={args.model}',
    f'model.length={args.length}',
    'parameterization=subs',
    'backbone=dit',
    f'trainer.accelerator={args.device}',
    'trainer.devices=1',
    f'sampling.steps={args.steps}',
    f'eval.checkpoint_path={args.checkpoint}',
    f'data.cache_dir={args.data_cache}',
    f'data.data_file={os.path.abspath(args.data_file)}',
    'data.emoji_vocab_sources=[text2emoji,common]',
    f'data.emoji_vocab_extra_files=[{os.path.abspath(args.data_file)}]',
    'data.emoji_include_challenge_in_train=false',
  ])
  if args.vocab_cache:
    config.data.emoji_vocab_cache = args.vocab_cache

  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    args.checkpoint, tokenizer=tokenizer, config=config)
  model.to(args.device)
  model.eval()

  prefix_ids, prefix_lens = [], []
  for emoji_prompt in prompts:
    prompt_ids = tokenizer(
      emoji_prompt, add_special_tokens=False)['input_ids']
    if not prompt_ids:
      print(f'Warning: no supported emoji in prompt {emoji_prompt!r}; '
            'using an empty prompt.')
    if tokenizer.unk_token_id in prompt_ids:
      print(f'Warning: prompt {emoji_prompt!r} contains emoji outside '
            'the atomic vocab; they will decode as [UNK_EMOJI].')
    prompt_ids = prompt_ids[:args.length - 2]
    prefix = (
      [tokenizer.bos_token_id]
      + prompt_ids
      + [tokenizer.sep_token_id])
    prefix_ids.append(prefix)
    prefix_lens.append(len(prefix))

  samples = model.restore_model_and_cond_sample(
    prefix_ids,
    num_steps=args.steps,
    max_response_tokens=args.max_response_tokens)
  samples = samples.cpu().tolist()

  print('\n=== Emoji generations ===')
  for emoji_prompt, ids, plen in zip(prompts, samples, prefix_lens):
    emoji_ids = extract_emoji(
      ids, plen, tokenizer.eos_token_id, tokenizer.pad_token_id)
    emoji = tokenizer.decode(emoji_ids).strip()
    print(f'{emoji_prompt!r:30s} -> {emoji}')


if __name__ == '__main__':
  main()
