"""Interactive emoji-diffusion chat server: send emoji, get an emoji reply.

Serves a chat GUI at / and three endpoints:

  GET  /vocab   the model's atomic emoji vocabulary, grouped for the picker
  GET  /health  readiness + backend mode
  POST /reply   run the MDLM conditional sampler (emoji prompt -> emoji reply)

Same-origin, so no CORS.

  python scripts/serve_emoji.py --checkpoint <ckpt> --port 8000

Reach it from a laptop with:  ssh -fN -L 8000:localhost:8000 primeintellect
then open http://localhost:8000

To work on the interface without a GPU or checkpoint, run the stub backend:

  python scripts/serve_emoji.py --mock

Mock mode serves the identical page and vocabulary shape but returns canned
replies. It never loads torch, so it runs anywhere fastapi is installed. Its
replies are NOT model output and are labelled as such in the UI.
"""
import argparse
import collections
import hashlib
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

import emoji_tokenization

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'static', 'emoji_chat.html')

app = FastAPI()
STATE = {'mode': 'mock'}


# --------------------------------------------------------------------------
# Vocabulary grouping for the picker.
#
# The picker offers only emoji the model actually has tokens for: anything
# outside the atomic vocabulary collapses to [UNK_EMOJI] and the model cannot
# condition on it. Grouping is by leading codepoint, which is coarse but needs
# no emoji-metadata dependency; anything unmatched lands in "More".
# --------------------------------------------------------------------------
# Codepoints whose neighbours belong to a different bucket than they do, so a
# pure range table would misfile them. Checked before the ranges.
_SINGLES = {
  0x1F48B: 'Smileys',  # kiss mark
  0x1F48F: 'People',   # kiss
  0x1F491: 'People',   # couple with heart
  0x1F490: 'Nature',   # bouquet
  0x1F48A: 'Objects',  # pill
  0x1F48D: 'Objects',  # ring
  0x1F454: 'Objects',  # necktie
  0x2B50: 'Symbols',   # star
  0x2614: 'Nature',    # umbrella with rain
  0x26F0: 'Nature',    # mountain
  0x1F54A: 'Nature',   # dove
  0x1F3C5: 'Activity',  # sports medal
  0x2328: 'Objects',   # keyboard
  0x203C: 'Symbols',   # double exclamation
  0x267E: 'Symbols',   # infinity
  0x1F9E0: 'People',   # brain
  0x1F9E1: 'Hearts',   # orange heart
}

# First match wins, so order these from most to least specific.
_RANGES = [
  ('Smileys', [(0x1F600, 0x1F644), (0x1F910, 0x1F93A), (0x1F970, 0x1F97A),
               (0x2639, 0x263A), (0x1FAE0, 0x1FAEF)]),
  ('Hearts', [(0x2763, 0x2764), (0x1F493, 0x1F49F), (0x1F5A4, 0x1F5A4),
              (0x1F90D, 0x1F90E)]),
  ('People', [(0x1F440, 0x1F450), (0x1F464, 0x1F487), (0x1F645, 0x1F64F),
              (0x1F926, 0x1F937), (0x1F918, 0x1F91F), (0x1F9B0, 0x1F9BF),
              (0x1F9D1, 0x1F9DF), (0x270A, 0x270D), (0x1FAC0, 0x1FACF),
              (0x1FAF0, 0x1FAFF)]),
  ('Food', [(0x1F32D, 0x1F37F), (0x1F940, 0x1F944), (0x1F950, 0x1F96F),
            (0x1F9C0, 0x1F9CB), (0x1FAD0, 0x1FADF), (0x2615, 0x2615)]),
  ('Nature', [(0x1F300, 0x1F335), (0x1F337, 0x1F32C), (0x1F400, 0x1F43F),
              (0x1F980, 0x1F9AF), (0x1FAB0, 0x1FABF), (0x2600, 0x2604),
              (0x26C4, 0x26C8), (0x2744, 0x2747), (0x1F308, 0x1F30F),
              (0x1F577, 0x1F578)]),
  ('Activity', [(0x1F380, 0x1F3C4), (0x1F3C6, 0x1F3FA), (0x26BD, 0x26BE),
                (0x1F93C, 0x1F94F), (0x1F3AE, 0x1F3B7), (0x1FA80, 0x1FA8F),
                (0x1F579, 0x1F57A), (0x26F7, 0x26F9)]),
  ('Travel', [(0x1F680, 0x1F6C5), (0x1F6CB, 0x1F6FF), (0x1F3E0, 0x1F3F0),
              (0x2708, 0x2708), (0x1F5FA, 0x1F5FF)]),
  ('Objects', [(0x1F4A0, 0x1F4FF), (0x1F500, 0x1F53D), (0x1F550, 0x1F567),
               (0x1F56F, 0x1F58F), (0x1F5A5, 0x1F5F9), (0x1F9E0, 0x1F9FF),
               (0x1FA70, 0x1FA7F), (0x1F97E, 0x1F97F), (0x231A, 0x231B),
               (0x23F0, 0x23F3), (0x1FA90, 0x1FA9F), (0x1FAA0, 0x1FAAF)]),
  ('Symbols', [(0x2705, 0x274E), (0x2757, 0x2762), (0x27A1, 0x27BF),
               (0x2795, 0x2797), (0x2B05, 0x2B07), (0x1F51F, 0x1F53D),
               (0x2049, 0x2049), (0x26A0, 0x26A1), (0x1F6A8, 0x1F6A8),
               (0x2733, 0x2734), (0x2B55, 0x2B55), (0x267B, 0x267F)]),
]


def _group_for(emoji):
  """Bucket an emoji grapheme by its first codepoint."""
  if not emoji:
    return 'More'
  cp = ord(emoji[0])
  # Regional indicators and tag sequences are flags.
  if 0x1F1E6 <= cp <= 0x1F1FF or cp == 0x1F3F4:
    return 'Flags'
  if cp in _SINGLES:
    return _SINGLES[cp]
  for name, spans in _RANGES:
    for lo, hi in spans:
      if lo <= cp <= hi:
        return name
  return 'More'


def build_groups(emoji_list):
  """Group + order emoji for the picker. Returns [{name, emoji:[...]}]."""
  buckets = collections.defaultdict(list)
  for emoji in emoji_list:
    buckets[_group_for(emoji)].append(emoji)
  order = ['Smileys', 'Hearts', 'People', 'Nature', 'Food', 'Activity',
           'Travel', 'Objects', 'Symbols', 'Flags', 'More']
  groups = [{'name': name, 'emoji': buckets[name]}
            for name in order if buckets.get(name)]
  # "All" first makes the whole vocabulary browsable in one place.
  if groups:
    groups.insert(0, {'name': 'All', 'emoji': list(emoji_list)})
  return groups


def model_emoji_vocab():
  """Emoji tokens from the loaded tokenizer, minus the special tokens."""
  tok = STATE['tok']
  import dataloader
  special = set(dataloader.ATOMIC_EMOJI_SPECIAL_TOKENS)
  vocab = tok.get_vocab()
  # Sort by token id so the picker follows the model's own vocabulary order.
  return [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])
          if t not in special]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
class Req(BaseModel):
  prompt: str
  steps: int = 32
  cap: Optional[int] = None


@app.get('/', response_class=HTMLResponse)
def index():
  try:
    with open(PAGE_PATH, encoding='utf-8') as handle:
      return handle.read()
  except FileNotFoundError:
    return HTMLResponse(
      f'<h1>Missing UI</h1><p>Expected the page at <code>{PAGE_PATH}</code>.</p>',
      status_code=500)


@app.get('/health')
def health():
  return {'ready': STATE.get('ready', False), 'mode': STATE['mode']}


@app.get('/vocab')
def vocab():
  is_mock = STATE['mode'] == 'mock'
  emoji_list = STATE['vocab']
  return {
    'mock': is_mock,
    'model': STATE.get('model_name', 'emoji-mdlm'),
    'count': len(emoji_list),
    'groups': build_groups(emoji_list),
  }


@app.post('/reply')
def reply(r: Req):
  emoji = ''.join(emoji_tokenization.extract_emoji_graphemes(r.prompt))
  if len(emoji) < 1:
    return JSONResponse({'reply': '', 'note': 'type at least 1 emoji'})

  if STATE['mode'] == 'mock':
    return {'reply': _mock_reply(emoji, r.cap), 'note': 'mock — not model output'}

  torch = STATE['torch']
  eps = STATE['eps']
  tok, model, length = STATE['tok'], STATE['model'], STATE['length']
  prefix = eps.prefix_for_prompt(tok, emoji, length)
  with torch.no_grad():
    out = model.restore_model_and_cond_sample(
      [prefix], num_steps=r.steps,
      max_response_tokens=r.cap).cpu().tolist()[0]
  return {'reply': eps.extract_response(tok, out, len(prefix))}


def _mock_reply(emoji, cap):
  """Deterministic stand-in so the interface can be exercised without a GPU."""
  pool = STATE['vocab']
  tokens = emoji_tokenization.extract_emoji_graphemes(emoji)
  digest = hashlib.sha256(''.join(tokens).encode('utf-8')).digest()
  count = cap if cap else (digest[0] % 3) + 2
  return ''.join(pool[digest[i + 1] % len(pool)] for i in range(count))


# --------------------------------------------------------------------------
def load_model(a):
  """Import the heavy stack and restore the checkpoint. Real mode only."""
  import torch
  # torch>=2.6 defaults weights_only=True, which rejects the omegaconf config
  # stored in Lightning checkpoints (trusted, locally-produced files).
  original_load = torch.load
  torch.load = lambda *args, **kw: original_load(
    *args, **{**kw, 'weights_only': False})
  import hydra
  import eval_permutation_stability as eps

  a.checkpoint = os.path.abspath(a.checkpoint)
  a.reply_data_file = os.path.abspath(a.reply_data_file)
  a.challenge_file = os.path.abspath(a.challenge_file)
  with hydra.initialize(version_base=None, config_path='../configs'):
    config = eps.build_config(a, a.checkpoint)
  tok = eps.dataloader.get_tokenizer(config)
  model = eps.diffusion.Diffusion.load_from_checkpoint(
    a.checkpoint, tokenizer=tok, config=config)
  model.to(a.device).eval()

  STATE.update(mode='model', torch=torch, eps=eps, tok=tok, model=model,
               length=a.length, model_name=a.model, ready=True)
  STATE['vocab'] = model_emoji_vocab()


FALLBACK_VOCAB = [
  '😀', '😃', '😄', '😁', '😆', '😅', '😂', '🤣', '😊', '😇',
  '🙂', '🙃', '😉', '😍', '😘', '😋', '😎', '🤩', '🥳', '😔',
  '😢', '😭', '😡', '😱', '😴', '🤔', '🤗', '👍', '👎', '👏',
  '🙌', '🙏', '💪', '👀', '💯', '✨', '🔥', '⭐', '🌟', '💫',
  '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '💕', '💔',
  '☀️', '🌧️', '🌈', '⚡', '❄️', '🌊', '🌍', '🌙', '🍕', '🍔',
  '🍟', '🍎', '🍰', '☕', '🎂', '🎉', '🎁', '🏆', '⚽', '🏀',
  '🎮', '🎵', '🎬', '📚', '💡', '💻', '📱', '🚗', '✈️', '🏠',
  '🏫', '💼', '⏰', '✅', '❌', '⚠️', '🚨', '❓', '❗', '➡️',
  '⬅️', '⬆️', '⬇️', '🇺🇸', '🏳️‍🌈', '👨‍👩‍👧‍👦', '👋', '🤝',
]


def load_mock(a):
  """Stand up the interface with no torch and no checkpoint.

  Vocabulary is taken from the real atomic vocab file when one has been built,
  otherwise from the emoji actually present in the reply dataset, so the picker
  still reflects this project rather than a generic emoji keyboard.
  """
  import json

  emoji_list = []
  source = 'built-in list'

  if a.vocab_cache and os.path.exists(a.vocab_cache):
    try:
      with open(a.vocab_cache, encoding='utf-8') as handle:
        loaded = json.load(handle)
      tokens = loaded if isinstance(loaded, list) else list(loaded)
      emoji_list = [t for t in tokens if not t.startswith('[')]
      source = a.vocab_cache
    except (ValueError, OSError):
      emoji_list = []

  if not emoji_list and os.path.exists(a.reply_data_file):
    seen = {}
    with open(a.reply_data_file, encoding='utf-8') as handle:
      for line in handle:
        if not line.strip():
          continue
        try:
          row = json.loads(line)
        except ValueError:
          continue
        for key in ('input', 'output', 'instruction'):
          for token in emoji_tokenization.extract_emoji_graphemes(
              row.get(key) or ''):
            seen[token] = seen.get(token, 0) + 1
    # Frequency order keeps the most-used emoji at the front of the picker.
    emoji_list = [t for t, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
    source = a.reply_data_file

  if not emoji_list:
    emoji_list = list(FALLBACK_VOCAB)

  STATE.update(mode='mock', vocab=emoji_list, ready=True,
               model_name='mock backend')
  print(f'MOCK MODE - no model loaded. Vocabulary from: {source} '
        f'({len(emoji_list)} emoji)', flush=True)


def main():
  ap = argparse.ArgumentParser(
    description='Serve the Emoji Diffusion Model chat interface.')
  ap.add_argument('--checkpoint',
                  help='Phase-2 reply checkpoint. Required unless --mock.')
  ap.add_argument('--mock', action='store_true',
                  help='Serve the interface with a stub backend (no torch).')
  ap.add_argument('--port', type=int, default=8000)
  ap.add_argument('--host', default='127.0.0.1')
  ap.add_argument('--model', default='medium')
  ap.add_argument('--length', type=int, default=64)
  ap.add_argument('--device', default='cuda')
  ap.add_argument('--data-cache', default='/tmp/emoji_mdlm_two_phase_hard1')
  ap.add_argument('--vocab-cache',
                  default='/tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json')
  ap.add_argument('--reply-data-file', default='data/emoji_reply/emoji_reply.jsonl')
  ap.add_argument('--challenge-file',
                  default='data/emoji_reply/curated_permutation.jsonl')
  ap.add_argument('--backbone', default='dit')
  ap.add_argument('--parameterization', default='subs')
  ap.add_argument('--steps', type=int, default=32)
  a = ap.parse_args()

  if a.mock:
    load_mock(a)
  else:
    if not a.checkpoint:
      ap.error('--checkpoint is required (or pass --mock to run without a model)')
    load_model(a)
    print('MODEL READY on port', a.port, flush=True)

  print(f'Open http://{a.host}:{a.port}', flush=True)
  uvicorn.run(app, host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
  main()
