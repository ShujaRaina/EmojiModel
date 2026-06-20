"""Build frozen text-teacher artifacts for emoji semantic transfer.

This creates a `semantic_table.pt` from KomeijiForce/Text2Emoji with:
  - frozen text teacher embeddings;
  - atomic emoji token ids;
  - row metadata.

Optionally writes nearest-neighbor emoji pairs as JSONL for `data=emoji_reply`.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(
  os.path.dirname(os.path.abspath(__file__))))

import datasets
import torch
import torch.nn.functional as F
import transformers

import dataloader


def mean_pool(last_hidden_state, attention_mask):
  mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
  summed = (last_hidden_state * mask).sum(dim=1)
  counts = mask.sum(dim=1).clamp(min=1)
  return summed / counts


def batched(values, batch_size):
  for i in range(0, len(values), batch_size):
    yield values[i:i + batch_size]


def build_text_embeddings(texts, model_name, batch_size, device):
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
  model = transformers.AutoModel.from_pretrained(model_name).to(device)
  model.eval()
  embeddings = []
  with torch.no_grad():
    for batch_texts in batched(texts, batch_size):
      batch = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors='pt').to(device)
      output = model(**batch)
      pooled = mean_pool(output.last_hidden_state, batch['attention_mask'])
      embeddings.append(F.normalize(pooled.cpu(), dim=-1))
  return torch.cat(embeddings, dim=0)


def write_pairs(path, rows, embeddings, top_k):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  sims = embeddings @ embeddings.T
  sims.fill_diagonal_(-1)
  with open(path, 'w', encoding='utf-8') as f:
    for i, row in enumerate(rows):
      neighbors = torch.topk(
        sims[i], k=min(top_k, sims.shape[0] - 1)).indices
      for j in neighbors.tolist():
        pair = {
          'prompt_emoji': row['emoji'],
          'response_emoji': rows[j]['emoji'],
          'source_text': row['text'],
          'target_text': rows[j]['text'],
          'similarity': float(sims[i, j]),
        }
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cache-dir', default=os.path.expanduser('~/mdlm_data'))
  parser.add_argument('--output', default='emoji_semantic_table.pt')
  parser.add_argument('--pairs-output', default=None)
  parser.add_argument('--teacher-model',
                      default='BAAI/bge-small-en-v1.5')
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--top-k', type=int, default=3)
  parser.add_argument('--limit', type=int, default=None,
                      help='Optional row limit for fast first runs.')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  args = parser.parse_args()

  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
  raw = datasets.load_dataset(
    'KomeijiForce/Text2Emoji',
    split='train',
    cache_dir=args.cache_dir)
  if args.limit:
    raw = raw.select(range(min(args.limit, len(raw))))
  emoji_tokens = dataloader._sorted_emoji_vocab(
    list(raw['emoji']) + dataloader.COMMON_EMOJI_ALLOWLIST)
  emoji_tokenizer = dataloader.AtomicEmojiTokenizer(emoji_tokens)

  rows = []
  token_ids = []
  for idx, row in enumerate(raw):
    emoji = row.get('emoji') or ''
    rows.append({
      'text_id': idx,
      'topic': row.get('topic'),
      'text': row.get('text') or '',
      'emoji': emoji,
    })
    token_ids.append(
      emoji_tokenizer(emoji, add_special_tokens=False)['input_ids'])

  text_embeddings = build_text_embeddings(
    [row['text'] for row in rows],
    args.teacher_model,
    args.batch_size,
    args.device)
  torch.save({
    'teacher_model': args.teacher_model,
    'emoji_vocab': emoji_tokens,
    'emoji_token_ids': token_ids,
    'text_embedding': text_embeddings,
    'rows': rows,
  }, args.output)

  if args.pairs_output:
    write_pairs(args.pairs_output, rows, text_embeddings, args.top_k)


if __name__ == '__main__':
  main()
