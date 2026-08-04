import collections
import functools
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)

from emoji_tokenization import (
  extract_emoji_graphemes,
  is_emoji_grapheme,
  split_graphemes,
)


ATOMIC_EMOJI_SPECIAL_TOKENS = [
  '[PAD]',
  '[BOS]',
  '[EOS]',
  '[SEP]',
  '[MASK]',
  '[UNK_EMOJI]',
]

COMMON_EMOJI_ALLOWLIST = [
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


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


class AtomicEmojiTokenizer(transformers.PreTrainedTokenizer):
  """Emoji-only tokenizer where each grapheme cluster is one token."""

  def __init__(self, emoji_tokens=None, **kwargs):
    emoji_tokens = emoji_tokens or []
    special_vocab = {
      token: i for i, token in enumerate(ATOMIC_EMOJI_SPECIAL_TOKENS)}
    unique_emoji = []
    seen = set(special_vocab)
    for token in emoji_tokens:
      if not is_emoji_grapheme(token):
        continue
      if token in seen:
        continue
      unique_emoji.append(token)
      seen.add(token)
    self._vocab_str_to_int = {
      **special_vocab,
      **{
        token: i + len(special_vocab)
        for i, token in enumerate(unique_emoji)
      }}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    self.emoji_tokens = unique_emoji
    super().__init__(
      pad_token='[PAD]',
      bos_token='[BOS]',
      eos_token='[EOS]',
      sep_token='[SEP]',
      mask_token='[MASK]',
      unk_token='[UNK_EMOJI]',
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return extract_emoji_graphemes(text)

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK_EMOJI]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str.get(index, '[UNK_EMOJI]')

  def convert_tokens_to_string(self, tokens):
    return ''.join(
      token for token in tokens
      if token not in ATOMIC_EMOJI_SPECIAL_TOKENS)

  def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
      return [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]
    return (
      [self.bos_token_id]
      + list(token_ids_0)
      + [self.sep_token_id]
      + list(token_ids_1)
      + [self.eos_token_id])

  def get_special_tokens_mask(
      self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
    if already_has_special_tokens:
      special_ids = set(self.all_special_ids)
      return [1 if token_id in special_ids else 0
              for token_id in token_ids_0]
    if token_ids_1 is None:
      return [1] + ([0] * len(token_ids_0)) + [1]
    return (
      [1]
      + ([0] * len(token_ids_0))
      + [1]
      + ([0] * len(token_ids_1))
      + [1])

  def create_token_type_ids_from_sequences(
      self, token_ids_0, token_ids_1=None):
    return [0] * len(self.build_inputs_with_special_tokens(
      token_ids_0, token_ids_1))

  def get_vocab(self) -> typing.Dict[str, int]:
    return dict(self._vocab_str_to_int)

  def save_vocabulary(self, save_directory, filename_prefix=None):
    if not os.path.isdir(save_directory):
      os.makedirs(save_directory, exist_ok=True)
    prefix = filename_prefix + '-' if filename_prefix else ''
    vocab_file = os.path.join(save_directory, f'{prefix}emoji_vocab.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
      json.dump(self.emoji_tokens, f, ensure_ascii=False, indent=2)
    return (vocab_file,)


def _path_fingerprint(path):
  return hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]


def _repo_root():
  return os.path.dirname(os.path.abspath(__file__))


def _resolve_repo_path(path):
  if not path or '://' in str(path):
    return path
  path = str(path)
  if os.path.isabs(path) or utils.fsspec_exists(path):
    return path
  repo_path = os.path.join(_repo_root(), path)
  if utils.fsspec_exists(repo_path):
    return repo_path
  return path


def _sorted_emoji_vocab(emoji_strings, min_count=1, always_keep=()):
  """Emoji vocabulary, optionally dropping tokens seen too rarely to learn.

  Frequency was previously discarded, which suited curated synthetic pools
  where every emoji recurs often. Real corpora have a long tail: in a 3.5k-pair
  Bluesky sample, 26% of distinct emoji appear exactly once. Each such emoji
  gets a dedicated token trained on a single example, which it cannot learn,
  while still enlarging the output softmax. Requiring min_count>=3 there cuts
  the vocabulary by 43% and costs 2.7% of token coverage; the remainder falls
  back to [UNK_EMOJI].

  always_keep holds emoji that stay in the vocabulary regardless of frequency,
  used for the curated allowlist so a base vocabulary survives the threshold.
  """
  counts = collections.Counter()
  for text in emoji_strings:
    counts.update(extract_emoji_graphemes(text))
  protected = set()
  for text in always_keep:
    protected.update(extract_emoji_graphemes(text))
  vocab = {token for token, n in counts.items()
           if n >= min_count or token in protected}
  return sorted(vocab)


def _read_emoji_strings(path):
  path = _resolve_repo_path(path)
  if not path or not utils.fsspec_exists(path):
    return []
  with fsspec.open(path, 'r', encoding='utf-8') as f:
    if path.endswith('.jsonl'):
      rows = [json.loads(line) for line in f if line.strip()]
    elif path.endswith('.json'):
      payload = json.load(f)
      if isinstance(payload, dict):
        rows = payload.get('data', payload.get('rows', payload))
        if isinstance(rows, dict):
          rows = rows.values()
      else:
        rows = payload
    else:
      return [line.strip() for line in f if line.strip()]
  strings = []
  for row in rows:
    if isinstance(row, dict):
      for key in (
          'emoji', 'output', 'response', 'response_emoji', 'target',
          'target_emoji', 'prompt', 'prompt_emoji', 'instruction', 'input'):
        if key in row and row[key] is not None:
          strings.append(str(row[key]))
    else:
      strings.append(str(row))
  return strings


def _read_emoji_vocab_cache(path):
  with fsspec.open(path, 'r', encoding='utf-8') as f:
    payload = json.load(f)
  if isinstance(payload, dict):
    return payload.get('emoji_tokens', [])
  return payload


def _load_emoji_reply_dataset(path, cache_dir):
  path = _resolve_repo_path(path)
  ext = os.path.splitext(path)[1].lower()
  if ext in {'.json', '.jsonl'}:
    return datasets.load_dataset(
      'json', data_files=path, split='train', cache_dir=cache_dir)
  if ext == '.csv':
    return datasets.load_dataset(
      'csv', data_files=path, split='train', cache_dir=cache_dir)
  if ext in {'.tsv', '.txt'}:
    return datasets.load_dataset(
      'csv', data_files=path, delimiter='\t',
      split='train', cache_dir=cache_dir)
  raise ValueError(
    'Unsupported emoji_reply data_file. Use JSON, JSONL, CSV, or TSV.')


def _normalize_emoji_reply_columns(raw):
  rename = {}
  if 'instruction' in raw.column_names and 'text' not in raw.column_names:
    rename['instruction'] = 'text'
  if 'output' in raw.column_names and 'emoji' not in raw.column_names:
    rename['output'] = 'emoji'
  if rename:
    raw = raw.rename_columns(rename)
  return raw


def _emoji_reply_row_fingerprint(row):
  payload = {
    'instruction': row.get('instruction') or row.get('text') or '',
    'input': (
      row.get('input')
      or row.get('prompt_emoji')
      or row.get('source_emoji')
      or ''),
    'output': (
      row.get('output')
      or row.get('emoji')
      or row.get('target_emoji')
      or row.get('response_emoji')
      or ''),
    'topic': row.get('topic') or '',
  }
  return hashlib.sha1(
    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
      'utf-8')).hexdigest()


def _load_emoji_reply_benchmark(path, cache_dir):
  path = _resolve_repo_path(path)
  if not path or not utils.fsspec_exists(path):
    return None
  return _normalize_emoji_reply_columns(
    _load_emoji_reply_dataset(path, cache_dir))


def _canonical_emoji_bag(text):
  tokens = extract_emoji_graphemes(str(text or ''))
  return ' '.join(sorted(tokens))


def _first_emoji_row_value(row, names):
  for name in names:
    if name not in row or row[name] is None:
      continue
    value = str(row[name])
    if extract_emoji_graphemes(value):
      return value
  return ''


def _emoji_reply_split_key(row, strategy):
  prompt = _first_emoji_row_value(row, [
    'prompt_emoji', 'source_emoji', 'input_emoji',
    'prompt', 'source', 'input', 'instruction', 'text'])
  response = _first_emoji_row_value(row, [
    'response_emoji', 'target_emoji', 'output_emoji',
    'response', 'target', 'output', 'emoji'])
  prompt_key = _canonical_emoji_bag(prompt)
  response_key = _canonical_emoji_bag(response)
  if strategy == 'prompt_bag':
    return prompt_key
  if strategy == 'response_bag':
    return response_key
  if strategy == 'prompt_response_bag':
    return f'{prompt_key}=>{response_key}'
  raise ValueError(
    'Unsupported emoji_split_strategy. Use random, prompt_bag, '
    'response_bag, or prompt_response_bag.')


def _split_emoji_reply_dataset(raw, data_config):
  validation_size = float(data_config.get('emoji_validation_size', 0.05))
  split_seed = int(data_config.get('emoji_split_seed', 42))
  strategy = data_config.get('emoji_split_strategy', 'random')
  benchmark_file = data_config.get('emoji_benchmark_file', None)
  benchmark = _load_emoji_reply_benchmark(
    benchmark_file, data_config.get('cache_dir', None))
  if benchmark is not None and len(benchmark):
    benchmark_fingerprints = {
      _emoji_reply_row_fingerprint(benchmark[i])
      for i in range(len(benchmark))}
    kept_indices = [
      idx for idx in range(len(raw))
      if _emoji_reply_row_fingerprint(raw[idx]) not in benchmark_fingerprints]
    held_out = len(raw) - len(kept_indices)
    if held_out:
      LOGGER.info(
        f'Emoji reply final benchmark holdout: {held_out} source rows '
        f'excluded before dev split; {len(benchmark)} benchmark rows are '
        'reserved for final external evaluation only.')
      raw = raw.select(kept_indices)
    else:
      LOGGER.warning(
        f'emoji_benchmark_file={benchmark_file} had no overlap with the '
        'training data; falling back to configured validation split.')

  if strategy == 'random':
    return raw.train_test_split(test_size=validation_size, seed=split_seed)

  rows = [raw[i] for i in range(len(raw))]
  keys = [_emoji_reply_split_key(row, strategy) for row in rows]
  unique_keys = sorted(set(keys))
  if len(unique_keys) <= 1:
    LOGGER.warning(
      f'emoji_split_strategy={strategy} produced <=1 unique split keys; '
      'falling back to row-random split.')
    return raw.train_test_split(test_size=validation_size, seed=split_seed)

  def key_sort_value(key):
    digest = hashlib.sha1(f'{split_seed}:{key}'.encode('utf-8')).hexdigest()
    return digest

  shuffled_keys = sorted(unique_keys, key=key_sort_value)
  holdout_count = max(1, int(round(len(shuffled_keys) * validation_size)))
  holdout_count = min(holdout_count, len(shuffled_keys) - 1)
  holdout_keys = set(shuffled_keys[:holdout_count])
  train_indices = [
    idx for idx, key in enumerate(keys) if key not in holdout_keys]
  valid_indices = [
    idx for idx, key in enumerate(keys) if key in holdout_keys]
  LOGGER.info(
    f'Emoji reply {strategy} split: '
    f'{len(train_indices)} train rows, {len(valid_indices)} validation rows, '
    f'{len(unique_keys)} unique bags, {len(holdout_keys)} held out.')
  return datasets.DatasetDict({
    'train': raw.select(train_indices),
    'test': raw.select(valid_indices),
  })


def _default_emoji_reply_data_file():
  return os.path.join(_repo_root(), 'data', 'emoji_reply', 'emoji_reply.jsonl')


def _default_emoji_challenge_data_file():
  return os.path.join(
    _repo_root(), 'data', 'emoji_reply', 'curated_permutation.jsonl')


def _write_emoji_vocab_cache(path, emoji_tokens):
  parent = os.path.dirname(path)
  if parent:
    utils.fsspec_mkdirs(parent, exist_ok=True)
  with fsspec.open(path, 'w', encoding='utf-8') as f:
    json.dump({'emoji_tokens': emoji_tokens},
              f, ensure_ascii=False, indent=2)


def _atomic_vocab_cache_path(config):
  path = config.data.get('emoji_vocab_cache', None)
  if not path:
    path = os.path.join(config.data.cache_dir, 'atomic_emoji_vocab.json')
  return path


def _build_atomic_emoji_vocab(config):
  cache_path = _atomic_vocab_cache_path(config)
  if cache_path and utils.fsspec_exists(cache_path):
    LOGGER.info(f'Loading atomic emoji vocab from: {cache_path}')
    return _read_emoji_vocab_cache(cache_path)

  emoji_strings = []
  sources = config.data.get(
    'emoji_vocab_sources', ['text2emoji', 'data_file', 'common'])
  if 'text2emoji' in sources:
    try:
      raw = datasets.load_dataset(
        'KomeijiForce/Text2Emoji',
        split='train',
        cache_dir=config.data.cache_dir)
      emoji_strings.extend(raw['emoji'])
    except Exception as exc:
      LOGGER.warning(
        'Could not load KomeijiForce/Text2Emoji while building the '
        f'atomic emoji vocab; falling back to configured lists. {exc}')
  if 'data_file' in sources:
    data_file = config.data.get('data_file', None)
    if not data_file and config.data.get('train', None) == 'emoji_reply':
      data_file = _default_emoji_reply_data_file()
    data_file = _resolve_repo_path(data_file)
    emoji_strings.extend(_read_emoji_strings(data_file))
  if 'curated' in sources:
    challenge_path = (
      config.data.get('emoji_challenge_set_path', None)
      or _default_emoji_challenge_data_file())
    challenge_path = _resolve_repo_path(challenge_path)
    emoji_strings.extend(_read_emoji_strings(challenge_path))
  if 'common' in sources:
    emoji_strings.extend(COMMON_EMOJI_ALLOWLIST)
  for extra_file in config.data.get('emoji_vocab_extra_files', []):
    emoji_strings.extend(_read_emoji_strings(extra_file))

  # The curated allowlist is a deliberate base vocabulary, so it is exempt from
  # the frequency threshold when it was requested as a source.
  always_keep = COMMON_EMOJI_ALLOWLIST if 'common' in sources else ()
  emoji_tokens = _sorted_emoji_vocab(
    emoji_strings,
    min_count=int(config.data.get('emoji_vocab_min_count', 1) or 1),
    always_keep=always_keep)
  if not emoji_tokens:
    raise ValueError(
      'Atomic emoji vocab is empty. Provide Text2Emoji, data.data_file, '
      'data.emoji_challenge_set_path, or include common in '
      'data.emoji_vocab_sources.')
  if cache_path:
    LOGGER.info(f'Writing atomic emoji vocab to: {cache_path}')
    _write_emoji_vocab_cache(cache_path, emoji_tokens)
  return emoji_tokens


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def _pad_tokenized_sequence(seq, pad_id, block_size):
  seq = seq[:block_size]
  attention = [1] * len(seq)
  if len(seq) < block_size:
    pad = block_size - len(seq)
    seq = seq + [pad_id] * pad
    attention = attention + [0] * pad
  return seq, attention


def _first_present(example, names):
  for name in names:
    if name in example:
      return example[name]
  return None


def _first_emoji_value(example, names, index):
  for name in names:
    if name not in example:
      continue
    value = example[name][index]
    if value is None:
      continue
    value = str(value)
    if extract_emoji_graphemes(value):
      return value
  return None


def _tokenize_text2emoji(example, tokenizer, block_size, eot=None,
                         paired=False,
                         max_emoji_tokens=32,
                         max_prompt_emoji_tokens=None,
                         max_response_emoji_tokens=None,
                         supervised_pad_tokens=0):
  """Build fixed-length emoji-only training sequences."""
  if max_prompt_emoji_tokens is None:
    max_prompt_emoji_tokens = max_emoji_tokens
  if max_response_emoji_tokens is None:
    max_response_emoji_tokens = max_emoji_tokens
  supervised_pad_tokens = max(0, int(supervised_pad_tokens or 0))
  input_ids_batch = []
  attention_batch = []
  cond_batch = []
  prompt_columns = [
    'prompt_emoji', 'source_emoji', 'input_emoji',
    'prompt', 'source', 'input', 'instruction', 'text']
  response_columns = [
    'response_emoji', 'target_emoji', 'output_emoji',
    'response', 'target', 'output', 'emoji']

  if paired:
    size = len(next(iter(example.values())))
    rows = (
      (_first_emoji_value(example, prompt_columns, i),
       _first_emoji_value(example, response_columns, i))
      for i in range(size))
  else:
    rows = ((emoji, None) for emoji in example['emoji'])

  for left_emoji, right_emoji in rows:
    if paired and not left_emoji and not right_emoji:
      continue
    left_emoji = left_emoji if left_emoji is not None else ''
    left_ids = tokenizer(
      left_emoji, add_special_tokens=False)['input_ids']
    left_ids = left_ids[:max_prompt_emoji_tokens]
    if right_emoji is None:
      seq = [tokenizer.bos_token_id] + left_ids + [tokenizer.eos_token_id]
      cond = [0] * len(seq)
    else:
      right_emoji = right_emoji if right_emoji is not None else ''
      right_ids = tokenizer(
        right_emoji, add_special_tokens=False)['input_ids']
      right_ids = right_ids[:max_response_emoji_tokens]
      supervised_pad = min(supervised_pad_tokens, max(0, block_size - 3))
      room_for_right = max(0, block_size - 3 - supervised_pad)
      right_ids = right_ids[:room_for_right]
      avail_left = block_size - 3 - len(right_ids) - supervised_pad
      if avail_left < 0:
        right_ids = right_ids[:block_size - 3]
        avail_left = 0
      left_ids = left_ids[:avail_left]
      prefix = (
        [tokenizer.bos_token_id]
        + left_ids
        + [tokenizer.sep_token_id])
      seq = (
        prefix
        + right_ids
        + [tokenizer.eos_token_id]
        + ([tokenizer.pad_token_id] * supervised_pad))
      cond = [1] * min(len(prefix), block_size)
      cond = cond + [0] * (len(seq) - len(cond))
    seq, attention = _pad_tokenized_sequence(
      seq, tokenizer.pad_token_id, block_size)
    if len(cond) < block_size:
      cond = cond + [0] * (block_size - len(cond))
    else:
      cond = cond[:block_size]
    input_ids_batch.append(seq)
    attention_batch.append(attention)
    cond_batch.append(cond)
  return {'input_ids': input_ids_batch,
          'attention_mask': attention_batch,
          'cond_mask': cond_batch}


def _usable_cpu_count():
  # os.sched_getaffinity is Linux-only; macOS needs os.cpu_count(). This is a
  # default argument, so on macOS the AttributeError fired at import time and
  # made the whole module unimportable.
  if hasattr(os, 'sched_getaffinity'):
    return len(os.sched_getaffinity(0))
  return os.cpu_count() or 1


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=None, streaming=False,
    data_config=None):
  if num_proc is None:
    num_proc = _usable_cpu_count()
  data_config = data_config or {}
  tokenizer_tag = 'atomic_emoji' if isinstance(
    tokenizer, AtomicEmojiTokenizer) else re.sub(
      r'[^A-Za-z0-9_.-]+', '_',
      getattr(tokenizer, 'name_or_path', None) or type(tokenizer).__name__)
  if isinstance(tokenizer, AtomicEmojiTokenizer):
    tokenizer_tag = (
      f'{tokenizer_tag}_{data_config.get("emoji_cache_version", "v2")}')
  cache_dataset_name = dataset_name
  if dataset_name == 'emoji_reply' and data_config.get('data_file', None):
    data_file = _resolve_repo_path(data_config.get('data_file', None))
    cache_dataset_name = (
      f'{dataset_name}_{_path_fingerprint(data_file)}')
  if dataset_name == 'emoji_reply' and data_config.get(
      'emoji_include_challenge_in_train', False):
    challenge_path = (
      data_config.get('emoji_challenge_set_path', None)
      or _default_emoji_challenge_data_file())
    challenge_path = _resolve_repo_path(challenge_path)
    cache_dataset_name = (
      f'{cache_dataset_name}_challenge_{_path_fingerprint(challenge_path)}')
    repeat = data_config.get('emoji_challenge_repeat', 1)
    if repeat and int(repeat) != 1:
      cache_dataset_name = f'{cache_dataset_name}_r{int(repeat)}'
  if dataset_name == 'emoji_reply':
    benchmark_file = data_config.get('emoji_benchmark_file', None)
    if benchmark_file:
      benchmark_path = _resolve_repo_path(benchmark_file)
      if benchmark_path and utils.fsspec_exists(benchmark_path):
        cache_dataset_name = (
          f'{cache_dataset_name}_bench_{_path_fingerprint(benchmark_path)}')
    split_strategy = data_config.get('emoji_split_strategy', 'random')
    split_seed = int(data_config.get('emoji_split_seed', 42))
    validation_size = float(data_config.get('emoji_validation_size', 0.05))
    if split_strategy != 'random':
      val_tag = str(validation_size).replace('.', 'p')
      cache_dataset_name = (
        f'{cache_dataset_name}_split{split_strategy}'
        f'_val{val_tag}_seed{split_seed}')
    prompt_cap = data_config.get('emoji_max_prompt_tokens', None)
    response_cap = data_config.get('emoji_max_response_tokens', None)
    pad_loss = data_config.get('emoji_supervised_pad_tokens', 0)
    if prompt_cap is not None:
      cache_dataset_name = f'{cache_dataset_name}_pmax{int(prompt_cap)}'
    if response_cap is not None:
      cache_dataset_name = f'{cache_dataset_name}_rmax{int(response_cap)}'
    if pad_loss and int(pad_loss) > 0:
      cache_dataset_name = f'{cache_dataset_name}_padloss{int(pad_loss)}'
  if wrap:
    filename = (
      f'{cache_dataset_name}_{tokenizer_tag}_{mode}_bs{block_size}'
      '_wrapped.dat')
  else:
    filename = (
      f'{cache_dataset_name}_{tokenizer_tag}_{mode}_bs{block_size}'
      '_unwrapped.dat')
  _path = os.path.join(cache_dir, filename)
  
  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'text2emoji':
    # KomeijiForce/Text2Emoji ships a single `train` split; carve out a
    # deterministic validation set so we can monitor val loss.
    raw = datasets.load_dataset(
      'KomeijiForce/Text2Emoji',
      split='train',
      cache_dir=cache_dir)
    split = raw.train_test_split(test_size=0.01, seed=42)
    dataset = datasets.DatasetDict(
      {'train': split['train'], 'validation': split['test']})
  elif dataset_name == 'emoji_reply':
    # Local instruction-tuning set. Defaults to the synthetic set built by
    # scripts/build_emoji_reply_dataset.py, but `data.data_file` can point to
    # the user's JSON/JSONL/CSV/TSV file.
    data_file = (
      data_config.get('data_file', None) or _default_emoji_reply_data_file())
    data_file = _resolve_repo_path(data_file)
    raw = _normalize_emoji_reply_columns(
      _load_emoji_reply_dataset(data_file, cache_dir))
    if data_config.get('emoji_include_challenge_in_train', False):
      challenge_path = (
        data_config.get('emoji_challenge_set_path', None)
        or _default_emoji_challenge_data_file())
      challenge_path = _resolve_repo_path(challenge_path)
      if utils.fsspec_exists(challenge_path):
        challenge = _normalize_emoji_reply_columns(
          _load_emoji_reply_dataset(challenge_path, cache_dir))
        repeat = max(1, int(data_config.get('emoji_challenge_repeat', 1)))
        raw = datasets.concatenate_datasets([raw] + [challenge] * repeat)
      else:
        LOGGER.warning(
          'emoji_include_challenge_in_train is true, but no challenge '
          f'dataset exists at {challenge_path}.')
    split = _split_emoji_reply_dataset(raw, data_config)
    dataset = datasets.DatasetDict(
      {'train': split['train'], 'validation': split['test']})
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if dataset_name == 'text2emoji':
      return _tokenize_text2emoji(
        example, tokenizer, block_size, EOS, paired=False)
    if dataset_name == 'emoji_reply':
      return _tokenize_text2emoji(
        example,
        tokenizer,
        block_size,
        EOS,
        paired=True,
        max_prompt_emoji_tokens=data_config.get(
          'emoji_max_prompt_tokens', None),
        max_response_emoji_tokens=data_config.get(
          'emoji_max_response_tokens', None),
        supervised_pad_tokens=data_config.get(
          'emoji_supervised_pad_tokens', 0))
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    map_kwargs = {
      'batched': True,
      'load_from_cache_file': dataset_name not in {'text2emoji', 'emoji_reply'},
      'desc': 'Tokenizing',
    }
    if num_proc and num_proc > 1:
      map_kwargs['num_proc'] = num_proc
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      **map_kwargs)
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  elif dataset_name == 'text2emoji':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'emoji', 'topic'])
  elif dataset_name == 'emoji_reply':
    remove_columns = [
      column for column in tokenized_dataset.column_names
      if column not in {'input_ids', 'attention_mask', 'cond_mask'}]
    if remove_columns:
      tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    map_kwargs = {
      'batched': True,
      'load_from_cache_file': dataset_name not in {'text2emoji', 'emoji_reply'},
      'desc': 'Grouping',
    }
    if num_proc and num_proc > 1:
      map_kwargs['num_proc'] = num_proc
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      **map_kwargs)
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'atomic_emoji':
    tokenizer = AtomicEmojiTokenizer(_build_atomic_emoji_vocab(config))
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

def _shuffle_span(seq, start, end):
  if end - start <= 1:
    return
  perm = torch.randperm(end - start, device=seq.device) + start
  seq[start:end] = seq[perm]


def _choose_emoji_infill_reveals(length, reveal_fraction, pattern, device):
  if length <= 1:
    return []
  reveal_count = int(round(length * reveal_fraction))
  reveal_count = min(max(1, reveal_count), length - 1)
  if pattern == 'mixed':
    choices = ['prefix', 'suffix', 'scattered']
    pattern = choices[int(torch.randint(len(choices), (), device=device).item())]
  if pattern == 'prefix':
    return list(range(reveal_count))
  if pattern == 'suffix':
    return list(range(length - reveal_count, length))
  if pattern == 'scattered':
    return torch.randperm(length, device=device)[:reveal_count].tolist()
  raise ValueError(
    'emoji_infill_reveal_pattern must be prefix, suffix, scattered, or mixed.')


def _apply_emoji_infill_conditioning(batch, tokenizer, data_config):
  prob = float(data_config.get('emoji_infill_train_prob', 0.0) or 0.0)
  if prob <= 0:
    return batch
  if not isinstance(tokenizer, AtomicEmojiTokenizer):
    return batch
  input_ids = batch.get('input_ids', None)
  cond_mask = batch.get('cond_mask', None)
  attention_mask = batch.get('attention_mask', None)
  if input_ids is None or cond_mask is None or attention_mask is None:
    return batch

  reveal_fraction = float(
    data_config.get('emoji_infill_reveal_fraction', 0.5) or 0.5)
  reveal_fraction = min(max(reveal_fraction, 0.0), 1.0)
  pattern = data_config.get('emoji_infill_reveal_pattern', 'mixed')
  sep_id = tokenizer.sep_token_id
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id
  for i in range(input_ids.shape[0]):
    if torch.rand((), device=input_ids.device).item() >= prob:
      continue
    seq = input_ids[i]
    sep_positions = (seq == sep_id).nonzero(as_tuple=False).flatten()
    if not len(sep_positions):
      continue
    sep = int(sep_positions[0].item())
    tail = seq[sep + 1:]
    end_offsets = ((tail == eos_id) | (tail == pad_id)).nonzero(
      as_tuple=False).flatten()
    end = sep + 1 + int(end_offsets[0].item()) if len(end_offsets) else len(seq)
    response_len = end - (sep + 1)
    if response_len <= 1:
      continue
    reveal_offsets = _choose_emoji_infill_reveals(
      response_len, reveal_fraction, pattern, input_ids.device)
    for offset in reveal_offsets:
      pos = sep + 1 + int(offset)
      if attention_mask[i, pos]:
        cond_mask[i, pos] = 1
  return batch


def _apply_emoji_permutation_augmentation(batch, tokenizer, data_config):
  prob = float(data_config.get('permutation_augment_prob', 0.0) or 0.0)
  if prob <= 0:
    return batch
  if not isinstance(tokenizer, AtomicEmojiTokenizer):
    return batch
  input_ids = batch.get('input_ids', None)
  cond_mask = batch.get('cond_mask', None)
  if input_ids is None or cond_mask is None:
    return batch

  shuffle_prompt = data_config.get('permutation_augment_prompt', True)
  shuffle_response = data_config.get('permutation_augment_response', True)
  sep_id = tokenizer.sep_token_id
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id
  for i in range(input_ids.shape[0]):
    if torch.rand((), device=input_ids.device).item() >= prob:
      continue
    seq = input_ids[i]
    sep_positions = (seq == sep_id).nonzero(as_tuple=False).flatten()
    if not len(sep_positions):
      continue
    sep = int(sep_positions[0].item())
    if shuffle_prompt:
      _shuffle_span(seq, 1, sep)
    if shuffle_response:
      tail = seq[sep + 1:]
      end_offsets = ((tail == eos_id) | (tail == pad_id)).nonzero(
        as_tuple=False).flatten()
      end = sep + 1 + int(end_offsets[0].item()) if len(end_offsets) else len(seq)
      _shuffle_span(seq, sep + 1, end)
  return batch


def _emoji_collate_fn(tokenizer, data_config, train):
  default_collate = torch.utils.data.default_collate
  if not train:
    return default_collate

  def collate(examples):
    batch = default_collate(examples)
    batch = _apply_emoji_permutation_augmentation(
      batch, tokenizer, data_config)
    return _apply_emoji_infill_conditioning(batch, tokenizer, data_config)
  return collate


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  # On CPU-only machines `device_count()` is 0; treat as a single device so
  # the batch-size bookkeeping below stays consistent with the trainer.
  num_gpus = max(torch.cuda.device_count(), 1)
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      num_proc=config.loader.num_workers,
      data_config=config.data)
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      num_proc=config.loader.num_workers,
      streaming=False,
      data_config=config.data)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=config.loader.num_workers > 0,
      collate_fn=_emoji_collate_fn(tokenizer, config.data, train=True))
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      collate_fn=_emoji_collate_fn(tokenizer, config.data, train=False))
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
