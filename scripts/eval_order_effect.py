"""Order-effect eval for the MDLM emoji model: adds a resample control and
best-of-6 references on top of the existing permutation-stability machinery,
so MDLM's numbers are directly comparable to the frontier baseline.

  perm_stability     = pairwise bag-Jaccard among replies to PERMUTED prompts
  resample_stability = pairwise bag-Jaccard among replies to the SAME prompt
  order_effect       = resample_stability - perm_stability   (~0 => order-free)
  target_bag_jaccard = vs the single human reply (comparable to committed nums)
  target_best6       = vs the best of all human replies for that prompt
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_permutation_stability as eps
import emoji_metrics


def best6(pred, refs):
    return max((emoji_metrics.bag_jaccard(pred, r) for r in refs), default=0.0)


def mean(x):
    return sum(x) / len(x) if x else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--label', default='mdlm')
    p.add_argument('--challenge-file', default='data/emoji_reply/curated_permutation.jsonl')
    p.add_argument('--reply-data-file', default='data/emoji_reply/emoji_reply.jsonl')
    p.add_argument('--data-cache', default='/tmp/emoji_mdlm_two_phase_hard1')
    p.add_argument('--vocab-cache', default='/tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json')
    p.add_argument('--num-groups', type=int, default=24)
    p.add_argument('--permutations-per-group', type=int, default=3)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--model', default='medium')
    p.add_argument('--device', default='cuda')
    p.add_argument('--backbone', default='dit')
    p.add_argument('--parameterization', default='subs')
    p.add_argument('--max-response-tokens', type=int, default=None)
    p.add_argument('--jsonl-out', default=None)
    args = p.parse_args()
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.reply_data_file = os.path.abspath(args.reply_data_file)
    args.challenge_file = os.path.abspath(args.challenge_file)

    refs_by_input = collections.defaultdict(list)
    for line in open(args.reply_data_file, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        inp = ''.join(eps.dataloader.extract_emoji_graphemes(d.get('input') or ''))
        out = d.get('output') or ''
        if inp and out:
            refs_by_input[inp].append(out)

    base = eps.read_reply_data_groups(
        args.reply_data_file, args.num_groups,
        args.permutations_per_group, args.seed)

    combined = collections.OrderedDict()
    originals = {}
    for gid, rows in base.items():
        original = rows[0]['prompt_emoji']
        originals[gid] = original
        perm_rows = []
        for r in rows:
            rr = dict(r); rr['canonical_id'] = gid + '__perm'; perm_rows.append(rr)
        combined[gid + '__perm'] = perm_rows
        res_rows = []
        for _ in range(len(rows)):
            rr = dict(rows[0]); rr['prompt_emoji'] = original
            rr['canonical_id'] = gid + '__resample'; res_rows.append(rr)
        combined[gid + '__resample'] = res_rows

    outputs_by_group, records = eps.sample_model(
        args, args.label, args.checkpoint, combined)

    perm_stab, res_stab, t1, t6 = [], [], [], []
    for gid in base:
        perm_recs = outputs_by_group.get(gid + '__perm', [])
        res_recs = outputs_by_group.get(gid + '__resample', [])
        if len(perm_recs) >= 2:
            perm_stab.append(eps.pairwise_stability(perm_recs))
        if len(res_recs) >= 2:
            res_stab.append(eps.pairwise_stability(res_recs))
        refs = refs_by_input.get(originals[gid], [])
        for r in perm_recs:
            t1.append(r.get('target_bag_jaccard', 0.0))
            t6.append(best6(r['generated_emoji'], refs) if refs
                      else r.get('target_bag_jaccard', 0.0))

    ps, rs = mean(perm_stab), mean(res_stab)
    print('\n=== %s | groups=%d perms=%d steps=%d cap=%s ===' % (
        args.label, len(base), args.permutations_per_group, args.steps,
        args.max_response_tokens))
    print('perm_stability       %.4f' % ps)
    print('resample_stability   %.4f' % rs)
    print('order_effect         %.4f' % (rs - ps))
    print('target_bag_jaccard   %.4f' % mean(t1))
    print('target_best6_jaccard %.4f' % mean(t6))

    if args.jsonl_out:
        with open(args.jsonl_out, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'summary': {
                'label': args.label, 'perm_stability': ps,
                'resample_stability': rs, 'order_effect': rs - ps,
                'target_bag_jaccard': mean(t1),
                'target_best6_jaccard': mean(t6),
                'cap': args.max_response_tokens}}, ensure_ascii=False) + '\n')
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
