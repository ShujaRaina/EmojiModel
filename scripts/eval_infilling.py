"""Any-position infilling eval for the MDLM emoji model.

Reveals a subset of a target reply's emoji and asks the model to fill the rest,
under three reveal patterns:
  prefix    - first k emoji revealed   (an AR model CAN condition on this)
  suffix    - last  k emoji revealed   (an AR model CANNOT)
  scattered - random k positions       (an AR model CANNOT)

A bidirectional diffusion model should infill all three about equally well;
that flatness is the structural advantage. Metric: multiset bag-F1 between the
infilled (masked) positions and the ground-truth emoji that were there.
"""
import argparse, collections, json, math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
import eval_permutation_stability as eps
import emoji_metrics


def semantic_geodesic_distance(cosine):
    return math.acos(max(-1.0, min(1.0, cosine))) / math.pi


def semantic_embedding_distance(scorer, left, right):
    left_embedding = scorer.embed(left)
    right_embedding = scorer.embed(right)
    if left_embedding is None and right_embedding is None:
        return 0.0
    if left_embedding is None or right_embedding is None:
        return 2.0
    return float(torch.linalg.vector_norm(left_embedding - right_embedding).item())


@torch.no_grad()
def infill(model, x_init, fixed, num_steps, eps_=1e-5):
    x = x_init.clone()
    x_fixed = x_init.clone()
    ts = torch.linspace(1, eps_, num_steps + 1, device=model.device)
    dt = (1 - eps_) / num_steps
    cache = None
    for i in range(num_steps):
        t = ts[i] * torch.ones(x.shape[0], 1, device=model.device)
        if model.sampler == 'ddpm':
            x = model._ddpm_update(x, t, dt)
        elif model.sampler == 'ddpm_cache':
            cache, x_next = model._ddpm_caching_update(x, t, dt, p_x0=cache)
            if (not torch.allclose(x_next, x)) or model.time_conditioning:
                cache = None
            x = x_next
        else:
            x = model._analytic_update(x, t, dt)
        x = torch.where(fixed, x_fixed, x)
    if model.config.sampling.noise_removal:
        t = ts[-1] * torch.ones(x.shape[0], 1, device=model.device)
        if model.sampler == 'analytic':
            x = model._denoiser_update(x, t)
        else:
            cond = model.noise(t)[0]
            x = model.forward(x, cond).argmax(dim=-1)
        x = torch.where(fixed, x_fixed, x)
    return x


def reveal_indices(n, k, pattern, rng):
    idx = list(range(n))
    if pattern == 'prefix':
        return set(idx[:k])
    if pattern == 'suffix':
        return set(idx[n - k:])
    return set(rng.sample(idx, k))  # scattered


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--challenge-file', default='data/emoji_reply/curated_permutation.jsonl')
    p.add_argument('--eval-file', default='data/emoji_reply/benchmark.jsonl',
                   help='JSONL rows to evaluate. Defaults to the reserved final '
                        'benchmark, which is excluded from no-leak training runs.')
    p.add_argument('--reply-data-file', default=None,
                   help='Deprecated alias for --eval-file.')
    p.add_argument('--exclude-file', default=None,
                   help='Optional JSONL rows to remove before split/eval, e.g. '
                        'the final benchmark when evaluating emoji_reply.jsonl.')
    p.add_argument('--data-cache', default='/tmp/emoji_mdlm_two_phase_hard1')
    p.add_argument('--vocab-cache', default='/tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json')
    p.add_argument('--num-examples', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--split-strategy', default='random')
    p.add_argument('--split-partition', default='all')
    p.add_argument('--split-validation-size', type=float, default=0.05)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--model', default='medium')
    p.add_argument('--device', default='cuda')
    p.add_argument('--backbone', default='dit')
    p.add_argument('--parameterization', default='subs')
    p.add_argument('--max-response-tokens', default=None)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--jsonl-out', default=None)
    p.add_argument('--semantic-table', default=None,
                   help='Optional semantic_table.pt for centered emoji '
                        'embedding cosine against masked truth.')
    args = p.parse_args()
    args.checkpoint = os.path.abspath(args.checkpoint)
    if args.reply_data_file:
        args.eval_file = args.reply_data_file
    args.eval_file = os.path.abspath(args.eval_file)
    args.reply_data_file = args.eval_file
    if args.exclude_file:
        args.exclude_file = os.path.abspath(args.exclude_file)
    args.challenge_file = os.path.abspath(args.challenge_file)
    rng = random.Random(args.seed)

    with hydra.initialize(version_base=None, config_path='../configs'):
        config = eps.build_config(args, args.checkpoint)
    tok = eps.dataloader.get_tokenizer(config)
    model = eps.diffusion.Diffusion.load_from_checkpoint(
        args.checkpoint, tokenizer=tok, config=config)
    model.to(args.device).eval()
    semantic_scorer = (
        eps.EmojiSemanticScorer(args.semantic_table)
        if args.semantic_table else None)
    mask_id = model.mask_index
    bos, sep, eos, pad = (tok.bos_token_id, tok.sep_token_id,
                          tok.eos_token_id, tok.pad_token_id)

    # Build (prompt, reply) examples from the requested split partition. For
    # no-leak claims, use the default reserved benchmark or another split that
    # was excluded from training.
    all_rows = [(i, json.loads(l)) for i, l in
                enumerate(open(args.eval_file, encoding='utf-8'))
                if l.strip()]
    if args.exclude_file:
        exclude_rows = [json.loads(l) for l in
                        open(args.exclude_file, encoding='utf-8') if l.strip()]
        exclude_fingerprints = {
            eps.dataloader._emoji_reply_row_fingerprint(row)
            for row in exclude_rows}
        all_rows = [
            item for item in all_rows
            if eps.dataloader._emoji_reply_row_fingerprint(item[1])
            not in exclude_fingerprints]
    part = eps._partition_reply_rows(
        all_rows, args.split_strategy, args.split_validation_size,
        args.split_seed, args.split_partition)
    examples = []
    for _, d in part:
        prompt = ''.join(eps.dataloader.extract_emoji_graphemes(
            d.get('input') or d.get('prompt_emoji') or ''))
        reply = (
            d.get('output') or d.get('target_emoji')
            or d.get('response_emoji') or '')
        reply_ids = tok(reply, add_special_tokens=False)['input_ids']
        if len(prompt) >= 1 and len(reply_ids) >= 2:
            examples.append((prompt, reply, reply_ids, d.get('topic', ''),
                             d.get('benchmark_id', '')))
        if len(examples) >= args.num_examples:
            break

    patterns = ['prefix', 'suffix', 'scattered']
    rows = []
    batch_x, batch_fixed, batch_meta = [], [], []

    def flush_batch():
        nonlocal batch_x, batch_fixed, batch_meta, rows
        if not batch_x:
            return
        x = torch.stack(batch_x, dim=0)
        fixed = torch.stack(batch_fixed, dim=0)
        out = infill(model, x, fixed, args.steps).cpu().tolist()
        for row_index, meta in enumerate(batch_meta):
            pred_ids = [
                out[row_index][meta['plen'] + j]
                for j in range(meta['rlen'])
                if j not in meta['revealed']]
            pred = tok.decode(pred_ids).strip()
            truth = tok.decode(meta['masked_truth']).strip()
            precision, recall, f1 = emoji_metrics.bag_precision_recall_f1(
                pred, truth)
            metrics = emoji_metrics.score_pair(pred, truth)
            metrics.update({
                'bag_precision': precision,
                'bag_recall': recall,
                'bag_f1': f1,
            })
            if semantic_scorer is not None:
                semantic_cosine = semantic_scorer.similarity(pred, truth)
                metrics['semantic_target_cosine'] = semantic_cosine
                metrics['semantic_geodesic_distance'] = (
                    semantic_geodesic_distance(semantic_cosine))
                metrics['semantic_embedding_distance'] = (
                    semantic_embedding_distance(semantic_scorer, pred, truth))
            rows.append({
                'model': meta['model'],
                'example_id': meta['example_id'],
                'sample_idx': meta['sample_idx'],
                'pattern': meta['pattern'],
                'benchmark_id': meta['benchmark_id'],
                'topic': meta['topic'],
                'prompt': meta['prompt'],
                'reply': meta['reply'],
                'revealed_k': meta['k'],
                'reply_len': meta['rlen'],
                'infilled': pred,
                'masked_truth': truth,
                **metrics,
            })
        batch_x, batch_fixed, batch_meta = [], [], []

    for example_idx, (prompt, reply_str, reply_ids, topic, benchmark_id) in enumerate(examples):
        pre = [bos] + tok(prompt, add_special_tokens=False)['input_ids'][:args.length - 2] + [sep]
        plen, rlen = len(pre), len(reply_ids)
        if plen + rlen + 1 > args.length:
            continue
        k = max(1, rlen // 2)
        for pat in patterns:
            revealed = reveal_indices(rlen, k, pat, rng)
            masked_truth = []
            for j in range(rlen):
                if j not in revealed:
                    masked_truth.append(reply_ids[j])
            for sample_idx in range(args.samples):
                x = torch.full((1, args.length), mask_id, dtype=torch.long, device=args.device)
                fixed = torch.zeros((1, args.length), dtype=torch.bool, device=args.device)
                # clamp prefix
                for j, t in enumerate(pre):
                    x[0, j] = t; fixed[0, j] = True
                # reply slots: reveal subset, mask rest
                for j in range(rlen):
                    pos = plen + j
                    if j in revealed:
                        x[0, pos] = reply_ids[j]; fixed[0, pos] = True
                # clamp EOS + pad tail (fix the length)
                x[0, plen + rlen] = eos; fixed[0, plen + rlen] = True
                for pos in range(plen + rlen + 1, args.length):
                    x[0, pos] = pad; fixed[0, pos] = True
                batch_x.append(x[0])
                batch_fixed.append(fixed[0])
                batch_meta.append({
                    'model': 'mdlm',
                    'example_id': f'emoji_reply_val_{example_idx:05d}',
                    'sample_idx': sample_idx,
                    'pattern': pat,
                    'benchmark_id': benchmark_id,
                    'topic': topic,
                    'prompt': prompt,
                    'reply': reply_str,
                    'k': k,
                    'rlen': rlen,
                    'plen': plen,
                    'revealed': revealed,
                    'masked_truth': masked_truth,
                })
                if len(batch_x) >= args.batch_size:
                    flush_batch()
    flush_batch()

    print('\n=== infilling bag-F1 by reveal pattern (n=%d examples) ===' % len(examples))
    for pat in patterns:
        fs = [r['bag_f1'] for r in rows if r['pattern'] == pat]
        print('  %-10s  F1 %.4f   (n=%d)' % (pat, sum(fs) / len(fs) if fs else 0.0, len(fs)))
    if args.jsonl_out:
        out_dir = os.path.dirname(args.jsonl_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.jsonl_out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('wrote ->', args.jsonl_out)


if __name__ == '__main__':
    main()
