"""Order-sensitivity eval: frontier models on emoji replies.

Improves on frontier_perm_stability.py in three ways:

1. RESAMPLE CONTROL (the real thesis test). `perm_stability` alone conflates
   two things: replies changing because the *prompt order* changed, and
   replies changing because *sampling is stochastic*. We add `resample_stability`
   = agreement among K replies to the SAME (original) prompt. Then:
       order_effect = resample_stability - perm_stability
   ~0 means order does not destabilize the reply beyond ordinary sampling noise
   (i.e. the model treats the prompt as an order-invariant bag). >0 means order
   specifically matters to the model.

2. BEST-OF-6 REFERENCES. Every prompt has 6 distinct human replies; we score
   target overlap against the best-matching human reply, not just one. We also
   report single-target jaccard for direct comparison with committed MDLM nums.

3. FLAGSHIP MODELS via OpenRouter.

Metric + grapheme logic are exact replicas of scripts/emoji_metrics.py.
perm_stability / single-target use the SAME 24 groups / 3 perms as the
committed MDLM eval, so those columns are directly comparable.

Usage:
  python eval/order_eval.py \
      --models openai/gpt-5.1 anthropic/claude-opus-4.8 google/gemini-2.5-pro \
      --num-groups 24 --permutations-per-group 3 --temperature 1.0 \
      --jsonl-out eval/order_eval_results.jsonl
"""
import argparse
import collections
import itertools
import json
import os
import random

import regex


# ---- exact replicas of dataloader/emoji_metrics grapheme + bag logic ----
def split_graphemes(text):
    return regex.findall(r"\X", text or "")


def is_emoji_grapheme(g):
    if not g or g.isspace():
        return False
    return bool(
        regex.search(r"\p{Extended_Pictographic}", g)
        or regex.search(r"\p{Regional_Indicator}", g)
        or "⃣" in g
        or any("\U000E0020" <= ch <= "\U000E007F" for ch in g)
    )


def emoji_tokens(text):
    return [g for g in split_graphemes(text) if is_emoji_grapheme(g)]


def emoji_bag(text):
    return collections.Counter(emoji_tokens(text))


def bag_jaccard(pred, target):
    pb, tb = emoji_bag(pred), emoji_bag(target)
    if not pb and not tb:
        return 1.0
    keys = set(pb) | set(tb)
    union = sum(max(pb[k], tb[k]) for k in keys)
    overlap = sum(min(pb[k], tb[k]) for k in keys)
    return overlap / union if union else 0.0


def best_jaccard(pred, refs):
    return max((bag_jaccard(pred, r) for r in refs), default=0.0)


# ---- groups (mirror read_reply_data_groups; partition=all) + 6 refs ------
def read_groups(path, num_groups, perms, seed):
    # all human replies per prompt-emoji, for best-of-6 scoring
    refs_by_input = collections.defaultdict(list)
    rows_raw = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rows_raw.append(row)
        inp = "".join(emoji_tokens(row.get("input") or ""))
        out = "".join(emoji_tokens(row.get("output") or ""))
        if inp and out:
            refs_by_input[inp].append(out)

    rng = random.Random(seed)
    groups = collections.OrderedDict()
    for line_no, row in enumerate(rows_raw):
        if len(groups) >= num_groups:
            break
        prompt_tokens = emoji_tokens(row.get("input") or "")
        response = "".join(emoji_tokens(row.get("output") or ""))
        if len(prompt_tokens) < 2 or not response:
            continue
        original = "".join(prompt_tokens)
        variants, seen = [original], {original}
        attempts = 0
        while len(variants) < perms and attempts < 64:
            attempts += 1
            shuffled = list(prompt_tokens)
            rng.shuffle(shuffled)
            cand = "".join(shuffled)
            if cand not in seen:
                variants.append(cand)
                seen.add(cand)
        if len(variants) < 2:
            continue
        gid = f"reply_{line_no:05d}_{row.get('topic', 'untagged')}"
        groups[gid] = {
            "original": original,
            "variants": variants,
            "target": response,
            "refs": refs_by_input.get(original, [response]),
            "topic": row.get("topic", ""),
        }
    return groups


# ----------------------------- OpenRouter ---------------------------------
def load_env_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    for line in open(env):
        if line.strip().startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found")


SYSTEM = (
    "You are replying to a casual text message that is written in emoji. "
    "Reply with ONLY emoji and no words -- a short, natural emotional reaction "
    "of about 2 to 4 emoji."
)


def query(client, model, prompt, temperature, cap):
    kw = dict(model=model, max_tokens=60,
              messages=[{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}])
    try:
        r = client.chat.completions.create(temperature=temperature, **kw)
    except Exception:
        r = client.chat.completions.create(**kw)  # model may reject temperature
    toks = emoji_tokens(r.choices[0].message.content or "")
    return "".join(toks[:cap] if cap else toks)


def run_all(client, model, prompts, temperature, cap, workers=8):
    """Query a batch of prompts concurrently, preserving order. Failed calls
    return '' so one bad request doesn't sink the run."""
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        try:
            return query(client, model, p, temperature, cap)
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, prompts))


def pairwise(replies):
    pairs = [bag_jaccard(a, b)
             for a, b in itertools.combinations(replies, 2)]
    return sum(pairs) / len(pairs) if pairs else 1.0


def evaluate(client, model, groups, temperature, cap):
    # flatten every call into one concurrent batch
    tasks, spans = [], {}
    for gid, g in groups.items():
        k = len(g["variants"])
        start = len(tasks)
        tasks.extend(g["variants"])                 # perm set
        tasks.extend([g["original"]] * k)           # resample set
        spans[gid] = (start, k)
    replies = run_all(client, model, tasks, temperature, cap)

    perm, resample, tgt1, tgt6 = [], [], [], []
    gens = {}
    for gid, g in groups.items():
        start, k = spans[gid]
        perm_replies = replies[start:start + k]
        resample_replies = replies[start + k:start + 2 * k]
        perm.append(pairwise(perm_replies))
        resample.append(pairwise(resample_replies))
        for rep in perm_replies:
            tgt1.append(bag_jaccard(rep, g["target"]))
            tgt6.append(best_jaccard(rep, g["refs"]))
        gens[gid] = {"perm_replies": perm_replies,
                     "resample_replies": resample_replies,
                     "target": g["target"], "refs": g["refs"]}
    ps = sum(perm) / len(perm)
    rs = sum(resample) / len(resample)
    return {
        "perm_stability": ps,
        "resample_stability": rs,
        "order_effect": rs - ps,
        "target_bag_jaccard": sum(tgt1) / len(tgt1),
        "target_best6_jaccard": sum(tgt6) / len(tgt6),
        "generations": gens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["openai/gpt-5.1", "anthropic/claude-opus-4.8",
                             "google/gemini-2.5-pro"])
    ap.add_argument("--data", default="data/emoji_reply/emoji_reply.jsonl")
    ap.add_argument("--num-groups", type=int, default=24)
    ap.add_argument("--permutations-per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--jsonl-out", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=load_env_key(), timeout=45, max_retries=2)
    groups = read_groups(args.data, args.num_groups,
                         args.permutations_per_group, args.seed)
    print(f"{len(groups)} groups, temp={args.temperature}, cap={args.cap}\n")

    results = {}
    for model in args.models:
        print(f"querying {model} ...")
        try:
            results[model] = evaluate(client, model, groups,
                                      args.temperature, args.cap)
        except Exception as e:
            print(f"  !! skipped {model}: {e}")

    cols = ["perm_stability", "resample_stability", "order_effect",
            "target_bag_jaccard", "target_best6_jaccard"]
    print("\n| model | " + " | ".join(cols) + " |")
    print("|---" + "|---:" * len(cols) + "|")
    print("| MDLM hard1_last (capped-3)* | 0.4778 | n/a | n/a | 0.2561 | n/a |")
    print("| MDLM hard1_last (uncapped)* | 0.2330 | n/a | n/a | 0.2271 | n/a |")
    for model, r in results.items():
        print("| " + model + " | "
              + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
    print("\n* committed MDLM numbers (single-target, no resample control)")

    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for model, r in results.items():
                summary = {k: v for k, v in r.items() if k != "generations"}
                f.write(json.dumps({"model": model, "summary": summary},
                                   ensure_ascii=False) + "\n")
                for gid, g in r["generations"].items():
                    f.write(json.dumps({"model": model, "group": gid, **g},
                                       ensure_ascii=False) + "\n")
        print(f"wrote -> {args.jsonl_out}")


if __name__ == "__main__":
    main()
