"""Frontier-model baseline for the emoji permutation-stability eval.

Replicates the *exact* metric + group construction of
scripts/eval_permutation_stability.py (--source reply_data, default random
split == all rows), but instead of sampling a local MDLM checkpoint it queries
frontier models through OpenRouter. This gives the ChatGPT/Claude baseline that
the committed results (hard1_last etc.) are missing.

Metric definitions (copied from scripts/emoji_metrics.py, torch-free):
  perm_stability     = mean pairwise multiset-Jaccard among the replies to the
                       permuted prompts in a group, averaged over groups
  target_bag_jaccard = mean multiset-Jaccard(reply, target), averaged

Prompt = the emoji `input` (permuted); target = the emoji `output`.

Usage:
  python eval/frontier_perm_stability.py \
      --models openai/gpt-4o-mini anthropic/claude-3.5-haiku \
      --num-groups 24 --permutations-per-group 3 --temperature 0
"""
import argparse
import collections
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared with the MDLM side via emoji_tokenization (regex-only, no torch), so
# both sides of the comparison are scored by identical code.
from emoji_tokenization import (  # noqa: E402
    bag_jaccard,
    emoji_bag,
    extract_emoji_graphemes,
)


# ----- group construction (mirrors read_reply_data_groups, partition=all) --
def read_groups(path, num_groups, perms, seed):
    rng = random.Random(seed)
    groups = collections.OrderedDict()
    for line_no, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        if len(groups) >= num_groups:
            break
        row = json.loads(line)
        prompt_tokens = extract_emoji_graphemes(
            row.get("input") or row.get("prompt_emoji") or "")
        response = "".join(extract_emoji_graphemes(row.get("output") or ""))
        if len(prompt_tokens) < 2 or not response:
            continue
        variants, seen = [], set()
        original = "".join(prompt_tokens)
        variants.append(original)
        seen.add(original)
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
        groups[gid] = [
            {"prompt_emoji": p, "target": response, "topic": row.get("topic", "")}
            for p in variants
        ]
    return groups


# ----------------------------- OpenRouter ---------------------------------
def load_env_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found (env or .env)")


SYSTEM = (
    "You are replying to a casual text message that is written in emoji. "
    "Reply with ONLY emoji and no words -- a short, natural emotional reaction "
    "of about 2 to 4 emoji."
)


def query(client, model, prompt, temperature, cap):
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=40,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    text = r.choices[0].message.content or ""
    toks = extract_emoji_graphemes(text)
    if cap:
        toks = toks[:cap]
    return "".join(toks)


def evaluate(client, model, groups, temperature, cap):
    gen = {}
    for gid, rows in groups.items():
        gen[gid] = [
            {**row, "reply": query(client, model, row["prompt_emoji"],
                                   temperature, cap)}
            for row in rows
        ]
    perm = []
    targ = []
    for gid, rows in gen.items():
        replies = [r["reply"] for r in rows]
        if len(replies) >= 2:
            pair = [bag_jaccard(a, b)
                    for a, b in itertools.combinations(replies, 2)]
            perm.append(sum(pair) / len(pair))
        targ.extend(bag_jaccard(r["reply"], r["target"]) for r in rows)
    return {
        "perm_stability": sum(perm) / len(perm) if perm else 0.0,
        "target_bag_jaccard": sum(targ) / len(targ) if targ else 0.0,
        "generations": gen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["openai/gpt-4o-mini",
                             "anthropic/claude-3.5-haiku"])
    ap.add_argument("--data", default="data/emoji_reply/emoji_reply.jsonl")
    ap.add_argument("--num-groups", type=int, default=24)
    ap.add_argument("--permutations-per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cap", type=int, default=None,
                    help="truncate replies to first N emoji (match capped-3)")
    ap.add_argument("--jsonl-out", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=load_env_key())

    groups = read_groups(args.data, args.num_groups,
                         args.permutations_per_group, args.seed)
    print(f"built {len(groups)} permutation groups "
          f"(temperature={args.temperature}, cap={args.cap})\n")

    results = {}
    for model in args.models:
        print(f"querying {model} ...")
        try:
            results[model] = evaluate(client, model, groups,
                                      args.temperature, args.cap)
        except Exception as e:
            print(f"  !! skipped {model}: {e}")

    print("\n| model | perm_stability | target_bag_jaccard |")
    print("|---|---:|---:|")
    # reference numbers from committed HARD_RESULTS.md
    ref = [("MDLM hard1_last (uncapped)", 0.2330, 0.2271),
           ("MDLM hard1_last (capped-3)", 0.4778, 0.2561)]
    for name, ps, tj in ref:
        print(f"| {name} | {ps:.4f} | {tj:.4f} |")
    for model, r in results.items():
        print(f"| {model} | {r['perm_stability']:.4f} "
              f"| {r['target_bag_jaccard']:.4f} |")

    if args.jsonl_out:
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for model, r in results.items():
                for gid, rows in r["generations"].items():
                    for row in rows:
                        f.write(json.dumps(
                            {"model": model, "group": gid, **row},
                            ensure_ascii=False) + "\n")
        print(f"\nwrote per-sample generations -> {args.jsonl_out}")


if __name__ == "__main__":
    main()
