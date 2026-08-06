"""Inspect AI eval: emoji-diffusion vs. frontier models on emoji replies.

THESIS UNDER TEST (honest framing):
  An emoji reply is closer to a *set* than a sequence -- its information is
  largely order-invariant. A non-autoregressive (masked diffusion) model has
  the matching inductive bias, so it should reproduce the *human* emoji-usage
  distribution -- order-free and diverse -- better than an autoregressive
  frontier model that commits to a canonical order and mode-collapses.

We do NOT claim to beat frontier models on raw semantic aptness. The headline
metric (`human_fidelity`) is a relevance-GATED blend of the dimensions where
the structural advantage actually lives:
    relevance (gate)  x  [ set-F1 to humans, pure-emoji format,
                           output diversity, order-symmetry ]
Per-component numbers are all reported so nothing is hidden.

Run:
  python eval/build_split.py                 # once, builds the test set
  inspect eval eval/emoji_task.py \
      --model emoji_diffusion/path/to/last.ckpt -M steps=128 --epochs 8
  inspect eval eval/emoji_task.py \
      --model openai/gpt-4o-mini -T temperature=1.0 --epochs 8
  inspect view                               # compare logs
"""
import os
from collections import defaultdict
from itertools import combinations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, metric, scorer
from inspect_ai.solver import TaskState, generate

# Registers the `emoji_diffusion` provider on import (needed for CLI runs).
import emoji_provider  # noqa: F401
from emoji_segment import extract_emoji, pure_emoji_ratio

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "data", "emoji_test.jsonl")

# Headline weights -- transparent on purpose. The structural dimensions
# (diversity, order-symmetry, format) carry real weight because they ARE the
# claim; quality (set-F1) and relevance keep us honest.
W = {
    "set_f1": 0.20,
    "pure_emoji": 0.20,
    "diversity": 0.25,
    "order_symmetry": 0.25,
    "relevance": 0.10,
}


# ----------------------------- dataset ------------------------------------
def _record_to_sample(rec):
    return Sample(
        input=(
            "Reply to this text message using ONLY emoji "
            f"(no words):\n\n{rec['prompt']}"
        ),
        target=rec["references"],          # list of human emoji replies
        metadata={"topic": rec["topic"], "prompt": rec["prompt"]},
    )


# ----------------------------- helpers ------------------------------------
def _multiset_f1(pred, ref):
    """F1 over emoji multisets (order ignored -- the whole point)."""
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pc, rc = defaultdict(int), defaultdict(int)
    for e in pred:
        pc[e] += 1
    for e in ref:
        rc[e] += 1
    overlap = sum(min(pc[e], rc[e]) for e in pc)
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pred), overlap / len(ref)
    return 2 * prec * rec / (prec + rec)


def _best_set_f1(pred, references):
    refs = [extract_emoji(r) for r in references]
    return max((_multiset_f1(pred, r) for r in refs), default=0.0)


def _pairwise_order_bias(sequences):
    """Mean |P(a before b) - 0.5| * 2 over co-occurring emoji pairs.

    0 == order is free (exchangeable, human-like); 1 == fully committed
    canonical order (the autoregressive failure mode).
    """
    before = defaultdict(int)
    seen = defaultdict(int)
    for seq in sequences:
        idx = {}
        for i, e in enumerate(seq):
            idx.setdefault(e, i)        # first occurrence position
        uniq = list(idx)
        for a, b in combinations(sorted(uniq), 2):
            seen[(a, b)] += 1
            if idx[a] < idx[b]:
                before[(a, b)] += 1
    if not seen:
        return 0.0
    biases = [abs(before[p] / seen[p] - 0.5) * 2 for p in seen]
    return sum(biases) / len(biases)


# ----------------------------- metrics ------------------------------------
def _meta(s):
    """Metadata dict, robust to being handed SampleScore or bare Score."""
    sc = s.score if hasattr(s, "score") else s
    return sc.metadata or {}


def _mean_field(scores, key, default=0.0):
    v = [_meta(s).get(key, default) for s in scores if _meta(s)]
    return sum(v) / len(v) if v else 0.0


def _group_by_prompt(scores):
    g = defaultdict(list)
    for s in scores:
        md = _meta(s)
        g[md.get("prompt", id(s))].append(md)
    return g


@metric
def relevance_rate():
    def m(scores):
        return _mean_field(scores, "relevant", 1.0)
    return m


@metric
def set_f1_mean():
    def m(scores):
        return _mean_field(scores, "set_f1")
    return m


@metric
def pure_emoji_rate():
    def m(scores):
        return _mean_field(scores, "pure_emoji")
    return m


@metric
def diversity():
    """Distinct emoji-SETS produced per prompt across epochs (mode-collapse
    detector). 1.0 == every sample is a different set; low == repeats itself."""
    def m(scores):
        ratios = []
        for prompt, mds in _group_by_prompt(scores).items():
            sets = [frozenset(md.get("emojis", [])) for md in mds]
            if len(sets) < 2:
                continue
            ratios.append(len(set(sets)) / len(sets))
        return sum(ratios) / len(ratios) if ratios else 0.0
    return m


@metric
def order_symmetry():
    """1 - pairwise-order-bias, aggregated per prompt. Higher == more order-
    free == better match to the exchangeable structure of emoji replies."""
    def m(scores):
        vals = []
        for prompt, mds in _group_by_prompt(scores).items():
            seqs = [md.get("emojis", []) for md in mds]
            seqs = [s for s in seqs if len(s) >= 2]
            if len(seqs) < 2:
                continue
            vals.append(1.0 - _pairwise_order_bias(seqs))
        return sum(vals) / len(vals) if vals else 0.0
    return m


@metric
def human_fidelity():
    """THE headline number: relevance-gated blend of all dimensions."""
    def m(scores):
        mds = [_meta(s) for s in scores if _meta(s)]
        if not mds:
            return 0.0
        rel = sum(md.get("relevant", 1.0) for md in mds) / len(mds)
        f1 = sum(md.get("set_f1", 0.0) for md in mds) / len(mds)
        pe = sum(md.get("pure_emoji", 0.0) for md in mds) / len(mds)
        div = diversity()(scores)
        sym = order_symmetry()(scores)
        gate = rel  # relevance gates the quality-ish terms
        return (
            W["relevance"] * rel
            + W["set_f1"] * f1 * gate
            + W["pure_emoji"] * pe
            + W["diversity"] * div
            + W["order_symmetry"] * sym
        )
    return m


# ----------------------------- scorer -------------------------------------
@scorer(metrics=[
    human_fidelity(), order_symmetry(), diversity(),
    set_f1_mean(), pure_emoji_rate(), relevance_rate(),
])
def emoji_scorer(relevance_grader=None):
    grader = get_model(relevance_grader) if relevance_grader else None

    async def score(state: TaskState, target: Target):
        completion = state.output.completion or ""
        emojis = extract_emoji(completion)
        pe = pure_emoji_ratio(completion)
        f1 = _best_set_f1(emojis, list(target.target))

        relevant = 1.0
        if grader is not None and emojis:
            prompt = (state.metadata or {}).get("prompt", "")
            q = (
                "A user texted: \"%s\". Someone replied with only these "
                "emoji: %s\nIs that a plausible, on-topic emotional reaction "
                "to the message? Answer strictly YES or NO."
                % (prompt, " ".join(emojis))
            )
            r = await grader.generate(q)
            relevant = 1.0 if "yes" in r.completion.strip().lower()[:5] else 0.0
        elif grader is not None and not emojis:
            relevant = 0.0

        return Score(
            value=f1,                       # convenient per-sample scalar
            answer=" ".join(emojis),
            metadata={
                "prompt": (state.metadata or {}).get("prompt", ""),
                "topic": (state.metadata or {}).get("topic", ""),
                "emojis": emojis,
                "set_f1": f1,
                "pure_emoji": pe,
                "relevant": relevant,
                "n_emoji": len(emojis),
            },
        )
    return score


# ----------------------------- task ---------------------------------------
@task
def emoji_task(relevance_grader=None, temperature=1.0):
    return Task(
        dataset=json_dataset(TEST, _record_to_sample),
        solver=generate(temperature=temperature),
        scorer=emoji_scorer(relevance_grader=relevance_grader),
        # diversity / order metrics need multiple draws per prompt -> set
        # --epochs N on the CLI (e.g. 8).
    )
