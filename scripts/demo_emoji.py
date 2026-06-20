"""Interactive emoji-only demo for the Phase 2 reply model.

The demo loads the trained checkpoint once, then probes each prompt alongside
shuffled prompt variants so order sensitivity is visible during conversation.
"""
import argparse
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


def build_config(args):
  overrides = [
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
  ]
  if args.vocab_cache:
    overrides.append(f'data.emoji_vocab_cache={args.vocab_cache}')
  with hydra.initialize(version_base=None, config_path='../configs'):
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


def shuffled_variants(prompt, count, seed):
  rng = random.Random(seed)
  tokens = dataloader.extract_emoji_graphemes(prompt)
  variants = []
  seen = set()
  original = ''.join(tokens)
  if original:
    variants.append(original)
    seen.add(original)
  attempts = 0
  while len(variants) < count and len(tokens) > 1 and attempts < 64:
    attempts += 1
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    candidate = ''.join(shuffled)
    if candidate not in seen:
      variants.append(candidate)
      seen.add(candidate)
  return variants


@torch.no_grad()
def generate(model, tokenizer, prompts, length, steps, max_response_tokens):
  prefix_ids = [prefix_for_prompt(tokenizer, prompt, length)
                for prompt in prompts]
  prefix_lens = [len(prefix) for prefix in prefix_ids]
  sample_ids = model.restore_model_and_cond_sample(
    prefix_ids,
    num_steps=steps,
    max_response_tokens=max_response_tokens).cpu().tolist()
  return [
    extract_response(tokenizer, ids, prefix_len)
    for ids, prefix_len in zip(sample_ids, prefix_lens)
  ]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint',
                      default=('outputs/emoji_two_phase_h100_full3/'
                               'phase2_reply/checkpoints/best.ckpt'))
  parser.add_argument('--data-cache',
                      default='/tmp/emoji_mdlm_two_phase_full3')
  parser.add_argument('--vocab-cache',
                      default='/tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json')
  parser.add_argument('--data-file',
                      default='data/emoji_reply/emoji_reply.jsonl')
  parser.add_argument('--steps', type=int, default=32)
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--model', default='small')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  parser.add_argument('--variants', type=int, default=3)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--max-response-tokens', type=int, default=3,
                      help='Clamp EOS after this many generated emoji.')
  parser.add_argument('--prompt', action='append', default=[])
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  config = build_config(args)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    args.checkpoint, tokenizer=tokenizer, config=config)
  model.to(args.device)
  model.eval()

  prompts = args.prompt
  if prompts:
    for prompt in prompts:
      variants = shuffled_variants(prompt, args.variants, args.seed)
      outputs = generate(
        model, tokenizer, variants, args.length, args.steps,
        args.max_response_tokens)
      for variant, output in zip(variants, outputs):
        print(f'{variant} -> {output}')
    return

  print('Emoji-only demo. Submit an emoji prompt, or blank/ctrl-d to exit.')
  while True:
    try:
      prompt = input('emoji> ').strip()
    except EOFError:
      print()
      break
    if not prompt:
      break
    variants = shuffled_variants(prompt, args.variants, args.seed)
    if not variants:
      print('No supported emoji found.')
      continue
    outputs = generate(
      model, tokenizer, variants, args.length, args.steps,
      args.max_response_tokens)
    for variant, output in zip(variants, outputs):
      print(f'{variant} -> {output}')


if __name__ == '__main__':
  main()
