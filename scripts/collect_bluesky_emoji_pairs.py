"""Collect real emoji reply pairs from the public Bluesky firehose.

Why this exists
---------------
`scripts/build_emoji_reply_dataset.py` draws a prompt and its reply as two
independent samples from the same hand-authored topic pool, so no information
flows from a prompt to its reply and neither side carries order. Any
order-invariance a model shows on that data is a property of the generator.

This collector produces pairs where the reply is a real human response to that
specific prompt, which is the one thing the synthetic data cannot provide.

How it works
------------
1. Subscribe to Jetstream, the public Bluesky firehose. No authentication.
2. Keep `app.bsky.feed.post` creates that are replies and carry enough emoji.
3. Resolve the parent post: DID -> PDS via plc.directory, then
   `com.atproto.repo.getRecord` against that PDS directly. This deliberately
   avoids the appview HTTP API, which is not reachable from every network.
4. Keep the pair if the parent is also emoji-bearing, then write the emoji.

Privacy
-------
Only emoji sequences are written. No post text, no handles, no DIDs, no URIs,
no timestamps. The output cannot be traced back to an account, and the raw
posts are never persisted. Author DIDs are held in memory purely to cap how
many pairs any single account can contribute, and are discarded on exit.

Usage
-----
    python scripts/collect_bluesky_emoji_pairs.py --target 20000 \
        --out data/emoji_reply/bluesky_pairs.jsonl

Resumable: re-running appends and skips pairs already present.
"""
import argparse
import asyncio
import collections
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emoji_tokenization

try:
  import websockets
except ImportError:
  raise SystemExit(
    'This collector needs the `websockets` package: pip install websockets')

JETSTREAM = ('wss://jetstream2.us-east.bsky.network/subscribe'
             '?wantedCollections=app.bsky.feed.post')
PLC_DIRECTORY = 'https://plc.directory'
USER_AGENT = 'emoji-diffusion-research/0.1'


# --------------------------------------------------------------------------
# Text quality
# --------------------------------------------------------------------------
def emoji_of(text):
  return emoji_tokenization.extract_emoji_graphemes(text or '')


def strip_emoji(text):
  keep = []
  for grapheme in emoji_tokenization.split_graphemes(text or ''):
    if not emoji_tokenization.is_emoji_grapheme(grapheme):
      keep.append(grapheme)
  return ''.join(keep)


def looks_like_spam(text):
  """Reject promotional and thread-marker posts.

  The firehose carries a lot of engagement spam, and those posts use emoji as
  bullets and arrows rather than as expression, which is exactly the signal we
  do not want to learn.
  """
  lowered = (text or '').lower()
  if 'http://' in lowered or 'https://' in lowered:
    return True
  if lowered.count('#') >= 3:
    return True
  if lowered.count('@') >= 3:
    return True
  stripped = (text or '').lstrip()
  # Thread markers: "5/", "3/7", "1)" at the start of a post.
  head = stripped[:4]
  if head[:1].isdigit() and ('/' in head or ')' in head):
    return True
  for phrase in ('follow me', 'follow back', 'link in bio', 'giveaway',
                 'promo', 'discount code', 'buy now', 'dm me'):
    if phrase in lowered:
      return True
  return False


def acceptable(text, args):
  """Is this post emoji-bearing enough for the emoji alone to carry meaning?"""
  tokens = emoji_of(text)
  if len(tokens) < args.min_emoji or len(tokens) > args.max_emoji:
    return None
  if len(set(tokens)) < args.min_distinct:
    return None
  # A distinct-count floor alone still admits 😂😂😂😂😂🤣🤣🤣🤣🤣, which has two
  # distinct emoji and almost no information. Require variety proportional to
  # length instead.
  if len(set(tokens)) / len(tokens) < args.min_variety:
    return None
  if looks_like_spam(text):
    return None
  # A long post with two emoji is a text post with decoration; the emoji do not
  # stand in for its meaning. Keep posts where the words are few.
  residue = strip_emoji(text).strip()
  if len(residue) > args.max_text_chars:
    return None
  return tokens


# --------------------------------------------------------------------------
# Parent resolution
# --------------------------------------------------------------------------
class ParentResolver:
  """did -> PDS host, then fetch a single post record from that PDS."""

  def __init__(self, timeout=12):
    self.timeout = timeout
    self.pds_cache = {}
    self.stats = collections.Counter()

  def _get_json(self, url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=self.timeout) as response:
      return json.loads(response.read())

  def _pds_for(self, did):
    if did in self.pds_cache:
      self.stats['pds_cache_hit'] += 1
      return self.pds_cache[did]
    doc = self._get_json(f'{PLC_DIRECTORY}/{did}')
    endpoint = None
    for service in doc.get('service', []):
      if service.get('type') == 'AtprotoPersonalDataServer':
        endpoint = service.get('serviceEndpoint')
        break
    self.pds_cache[did] = endpoint
    return endpoint

  def parent_text(self, at_uri):
    """Return the parent post's text, or None if it cannot be fetched."""
    try:
      did, collection, rkey = at_uri.replace('at://', '').split('/')
    except ValueError:
      self.stats['bad_uri'] += 1
      return None
    if collection != 'app.bsky.feed.post':
      self.stats['not_a_post'] += 1
      return None
    try:
      pds = self._pds_for(did)
      if not pds:
        self.stats['no_pds'] += 1
        return None
      url = (f'{pds}/xrpc/com.atproto.repo.getRecord'
             f'?repo={did}&collection={collection}&rkey={rkey}')
      record = self._get_json(url)
      self.stats['resolved'] += 1
      return (record.get('value') or {}).get('text')
    except urllib.error.HTTPError as error:
      # 400/404 are ordinary: the post was deleted or the repo is gone.
      self.stats[f'http_{error.code}'] += 1
      return None
    except Exception as error:
      self.stats[type(error).__name__] += 1
      return None


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------
def load_existing(path):
  """Prior pairs and reply frequencies, so a resumed run keeps its caps."""
  seen = set()
  response_counts = collections.Counter()
  if not os.path.exists(path):
    return seen, response_counts
  with open(path, encoding='utf-8') as handle:
    for line in handle:
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except ValueError:
        continue
      seen.add((row.get('input', ''), row.get('output', '')))
      response_counts[row.get('output', '')] += 1
  return seen, response_counts


async def collect(args):
  os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
  seen, response_counts = load_existing(args.out)
  print(f'resuming with {len(seen)} existing pairs', flush=True)

  resolver = ParentResolver(timeout=args.http_timeout)
  per_author = collections.Counter()
  stats = collections.Counter()
  stop = {'now': False}
  started = time.time()

  def request_stop(*_):
    if stop['now']:
      raise KeyboardInterrupt
    stop['now'] = True
    print('\nfinishing current work, press Ctrl-C again to force', flush=True)

  signal.signal(signal.SIGINT, request_stop)

  # Bounded concurrency: the firehose is far faster than parent resolution, so
  # without a cap the pending-resolution set grows without limit.
  semaphore = asyncio.Semaphore(args.concurrency)
  out_lock = asyncio.Lock()
  handle = open(args.out, 'a', encoding='utf-8')

  async def handle_reply(reply_tokens, parent_uri, author_did):
    async with semaphore:
      parent_text = await asyncio.to_thread(resolver.parent_text, parent_uri)
    if not parent_text:
      stats['parent_unavailable'] += 1
      return
    parent_tokens = acceptable(parent_text, args)
    if parent_tokens is None:
      stats['parent_rejected'] += 1
      return

    prompt = ''.join(parent_tokens)
    response = ''.join(reply_tokens)
    # A reply that merely repeats the prompt carries no response signal — it is
    # usually a bot echoing a template. Genuine reciprocity scores far lower:
    # ☕☀️ -> 🤗☕ is ~0.33, 🧡🩷 -> 🩵❤️ is 0.
    if emoji_tokenization.bag_jaccard(prompt, response) > args.max_echo:
      stats['echo'] += 1
      return
    key = (prompt, response)
    async with out_lock:
      if key in seen:
        stats['duplicate'] += 1
        return
      if per_author[author_did] >= args.max_per_author:
        stats['author_capped'] += 1
        return
      # One account posting the same canned reply to many prompts would
      # otherwise teach the model a constant.
      if response_counts[response] >= args.max_per_response:
        stats['response_capped'] += 1
        return
      seen.add(key)
      per_author[author_did] += 1
      response_counts[response] += 1
      handle.write(json.dumps({
        'instruction': '',
        'input': prompt,
        'output': response,
        'topic': 'bluesky',
        'source': 'bluesky',
      }, ensure_ascii=False) + '\n')
      handle.flush()
      stats['kept'] += 1

  pending = set()
  try:
    while not stop['now'] and stats['kept'] < args.target:
      try:
        async with websockets.connect(JETSTREAM, open_timeout=30,
                                      max_queue=4096) as socket:
          print('connected to jetstream', flush=True)
          while not stop['now'] and stats['kept'] < args.target:
            raw = await asyncio.wait_for(socket.recv(), timeout=60)
            stats['messages'] += 1
            try:
              message = json.loads(raw)
            except ValueError:
              continue
            commit = message.get('commit') or {}
            if commit.get('operation') != 'create':
              continue
            record = commit.get('record') or {}
            if record.get('$type') != 'app.bsky.feed.post':
              continue
            stats['posts'] += 1
            reply_ref = record.get('reply')
            if not reply_ref:
              continue
            stats['replies'] += 1
            tokens = acceptable(record.get('text'), args)
            if tokens is None:
              continue
            stats['reply_candidates'] += 1
            parent_uri = (reply_ref.get('parent') or {}).get('uri')
            if not parent_uri:
              continue

            task = asyncio.create_task(
              handle_reply(tokens, parent_uri, message.get('did')))
            pending.add(task)
            task.add_done_callback(pending.discard)

            if stats['messages'] % args.log_every == 0:
              rate = stats['kept'] / max(1e-9, time.time() - started) * 3600
              print(f"  msgs={stats['messages']:>8}  replies={stats['replies']:>6}"
                    f"  candidates={stats['reply_candidates']:>5}"
                    f"  kept={stats['kept']:>5}  (~{rate:.0f}/hr)", flush=True)
      except (asyncio.TimeoutError, websockets.WebSocketException, OSError) as e:
        if stop['now']:
          break
        print(f'  stream error ({type(e).__name__}), reconnecting in 5s',
              flush=True)
        await asyncio.sleep(5)
  except KeyboardInterrupt:
    pass
  finally:
    if pending:
      print(f'draining {len(pending)} in-flight lookups', flush=True)
      await asyncio.gather(*pending, return_exceptions=True)
    handle.close()

  elapsed = time.time() - started
  print(f'\ncollected {stats["kept"]} pairs in {elapsed/60:.1f} min '
        f'-> {args.out}')
  print('  filter stats :', dict(stats))
  print('  resolver     :', dict(resolver.stats))
  print(f'  unique authors: {len(per_author)}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out', default='data/emoji_reply/bluesky_pairs.jsonl')
  parser.add_argument('--target', type=int, default=20000,
                      help='stop after this many kept pairs')
  parser.add_argument('--min-emoji', type=int, default=2,
                      help='minimum emoji tokens on each side')
  parser.add_argument('--max-emoji', type=int, default=12)
  parser.add_argument('--min-distinct', type=int, default=2,
                      help='reject 😂😂😂: needs this many distinct emoji')
  parser.add_argument('--min-variety', type=float, default=0.5,
                      help='minimum distinct/total ratio; rejects laugh-spam '
                           'like 😂😂😂😂😂🤣🤣🤣🤣🤣 that clears --min-distinct')
  parser.add_argument('--max-echo', type=float, default=0.8,
                      help='reject replies this similar to their prompt '
                           '(bag Jaccard); catches bots echoing a template')
  parser.add_argument('--max-text-chars', type=int, default=80,
                      help='max non-emoji characters; keeps emoji load-bearing')
  parser.add_argument('--max-per-author', type=int, default=25,
                      help='cap one account dominating the dataset')
  parser.add_argument('--max-per-response', type=int, default=3,
                      help='cap how often one identical reply may appear')
  parser.add_argument('--concurrency', type=int, default=8)
  parser.add_argument('--http-timeout', type=int, default=12)
  parser.add_argument('--log-every', type=int, default=20000)
  args = parser.parse_args()
  asyncio.run(collect(args))


if __name__ == '__main__':
  main()
