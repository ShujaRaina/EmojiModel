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
import argparse, collections, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
import eval_permutation_stability as eps
import emoji_metrics


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
    p.add_argument('--reply-data-file', default='data/emoji_reply/emoji_reply.jsonl')
    p.add_argument('--data-cache', default='/tmp/emoji_mdlm_two_phase_hard1')
    p.add_argument('--vocab-cache', default='/tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json')
    p.add_argument('--num-examples', type=int, default=40)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--model', default='medium')
    p.add_argument('--device', default='cuda')
    p.add_argument('--backbone', default='dit')
    p.add_argument('--parameterization', default='subs')
    p.add_argument('--max-response-tokens', default=None)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--jsonl-out', default=None)
    args = p.parse_args()
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.reply_data_file = os.path.abspath(args.reply_data_file)
    args.challenge_file = os.path.abspath(args.challenge_file)
    rng = random.Random(args.seed)

    with hydra.initialize(version_base=None, config_path='../configs'):
        config = eps.build_config(args, args.checkpoint)
    tok = eps.dataloader.get_tokenizer(config)
    model = eps.diffusion.Diffusion.load_from_checkpoint(
        args.checkpoint, tokenizer=tok, config=config)
    model.to(args.device).eval()
    mask_id = model.mask_index
    bos, sep, eos, pad = (tok.bos_token_id, tok.sep_token_id,
                          tok.eos_token_id, tok.pad_token_id)

    # build (prompt, reply) examples
    examples = []
    for line in open(args.reply_data_file, encoding='utf-8'):
        if not line.strip():
            continue
        d = json.loads(line)
        prompt = ''.join(eps.dataloader.extract_emoji_graphemes(d.get('input') or ''))
        reply_ids = tok(d.get('output') or '', add_special_tokens=False)['input_ids']
        if len(prompt) >= 1 and len(reply_ids) >= 2:
            examples.append((prompt, d.get('output'), reply_ids, d.get('topic', '')))
        if len(examples) >= args.num_examples:
            break

    patterns = ['prefix', 'suffix', 'scattered']
    rows = []
    for prompt, reply_str, reply_ids, topic in examples:
        pre = [bos] + tok(prompt, add_special_tokens=False)['input_ids'][:args.length - 2] + [sep]
        plen, rlen = len(pre), len(reply_ids)
        if plen + rlen + 1 > args.length:
            continue
        k = max(1, rlen // 2)
        for pat in patterns:
            revealed = reveal_indices(rlen, k, pat, rng)
            x = torch.full((1, args.length), mask_id, dtype=torch.long, device=args.device)
            fixed = torch.zeros((1, args.length), dtype=torch.bool, device=args.device)
            # clamp prefix
            for j, t in enumerate(pre):
                x[0, j] = t; fixed[0, j] = True
            # reply slots: reveal subset, mask rest
            masked_truth = []
            for j in range(rlen):
                pos = plen + j
                if j in revealed:
                    x[0, pos] = reply_ids[j]; fixed[0, pos] = True
                else:
                    masked_truth.append(reply_ids[j])
            # clamp EOS + pad tail (fix the length)
            x[0, plen + rlen] = eos; fixed[0, plen + rlen] = True
            for pos in range(plen + rlen + 1, args.length):
                x[0, pos] = pad; fixed[0, pos] = True
            out = infill(model, x, fixed, args.steps)[0].cpu().tolist()
            pred_ids = [out[plen + j] for j in range(rlen) if j not in revealed]
            pred = tok.decode(pred_ids).strip()
            truth = tok.decode(masked_truth).strip()
            f1 = emoji_metrics.bag_precision_recall_f1(pred, truth)[2]
            rows.append({'pattern': pat, 'topic': topic, 'prompt': prompt,
                         'reply': reply_str, 'revealed_k': k, 'reply_len': rlen,
                         'infilled': pred, 'masked_truth': truth, 'bag_f1': f1})

    print('\n=== infilling bag-F1 by reveal pattern (n=%d examples) ===' % len(examples))
    for pat in patterns:
        fs = [r['bag_f1'] for r in rows if r['pattern'] == pat]
        print('  %-10s  F1 %.4f   (n=%d)' % (pat, sum(fs) / len(fs) if fs else 0.0, len(fs)))
    if args.jsonl_out:
        with open(args.jsonl_out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('wrote ->', args.jsonl_out)


if __name__ == '__main__':
    main()
