# Emoji Diffusion

Emoji Diffusion is a masked-diffusion language model that replies in emoji. It
adapts MDLM to model emoji sequences with an atomic grapheme vocabulary, and
adds the machinery to test whether treating them as holistic semantic objects
beats left-to-right decoding.

The core demo: give the model an emoji prompt, generate an emoji-only reply,
then shuffle the prompt order and compare how stable the reply semantics are.

[W&B report: Emoji](https://wandb.ai/derektang-the-university-of-chicago/emoji-model/reports/Emoji--VmlldzoxNzI4OTc5NQ)

> **Read [Limitations](#limitations) before citing any number here.** The
> training data is synthetic and template-generated, which constrains what the
> results can support. That section says exactly what does and does not follow.

## The Hypothesis

Emoji messages may not be ordinary token sequences. These prompts arguably
carry similar intent even when their order changes:

```text
😊😡😢
😢🔥😊
😡😊😢
```

Autoregressive models make order the main structure. Masked diffusion sees the
whole expression while denoising, which should make it a better fit for emoji
semantics, permutation stability, and infilling.

That is the hypothesis this project was built to test. It is worth being clear
that the project **has not confirmed it** — see [Limitations](#limitations).
What the project does provide is the machinery to test it: an emoji-native
tokenizer, a conditional diffusion pipeline, and an evaluation harness.

This repo builds on [MDLM](https://github.com/kuleshov-group/mdlm), the
NeurIPS 2024 masked discrete diffusion language model, and adds:

- an atomic emoji tokenizer where each emoji grapheme cluster is one token
- a two-phase emoji training pipeline
- emoji-to-emoji conditional generation
- permutation-stability and infilling evaluations
- a small FastAPI demo for live generation
- OpenRouter frontier-model baselines for comparison

## Demo

Run the local web demo from a checkpoint:

The checkpoint path below assumes the local H100 run artifacts are available.
Model checkpoints are intentionally not committed to Git.

```bash
python scripts/serve_emoji.py \
  --checkpoint outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_full3 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json \
  --reply-data-file data/emoji_reply/emoji_reply.jsonl \
  --device cuda \
  --port 8000
```

Open `http://localhost:8000`.

For a terminal demo:

```bash
python scripts/demo_emoji.py \
  --checkpoint outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_full3 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json \
  --data-file data/emoji_reply/emoji_reply.jsonl \
  --steps 32 \
  --device cuda
```

## How It Works

Phase 1 learns semantic grounding from text supervision. A teacher embedding
model maps text prompts into a semantic vector space, and the emoji-side model
learns to align emoji expressions to that space.

Phase 2 trains emoji-to-emoji replies directly:

```text
[BOS] prompt_emoji [SEP] response_emoji [EOS]
```

At inference time, the prompt side is clamped and the response side starts as
masks:

```text
[BOS] 😢💔 [SEP] [MASK] [MASK] [MASK] [EOS]
```

The diffusion model iteratively fills the response:

```text
[BOS] 😢💔 [SEP] 🤗 ❤️ ✨ [EOS]
```

## Quickstart

CPU setup for data work, smoke tests, and small demos:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

GPU setup for training:

```bash
conda env create -f requirements.yaml
conda activate mdlm
mkdir -p outputs watch_folder
```

## Key Commands

Build the emoji-reply dataset:

```bash
python scripts/build_emoji_reply_dataset.py --per-message 8 --seed 42
```

Train the two-phase H100 pipeline:

```bash
bash scripts/run_two_phase_h100.sh
```

Train only the emoji-reply model:

```bash
bash scripts/train_emoji_reply.sh
```

Run permutation-stability evaluation:

```bash
python scripts/eval_permutation_stability.py \
  --checkpoint outputs/emoji_two_phase_h100_full3/phase2_reply/checkpoints/best.ckpt \
  --data-cache /tmp/emoji_mdlm_two_phase_full3 \
  --vocab-cache /tmp/emoji_mdlm_two_phase_full3/atomic_emoji_vocab.json \
  --reply-data-file data/emoji_reply/emoji_reply.jsonl \
  --challenge-file data/emoji_reply/curated_permutation.jsonl \
  --device cuda
```

Run emoji infilling evaluation:

```bash
python scripts/eval_infilling.py \
  --checkpoint outputs/emoji_phase2_infill_promptbag_h100_long1/checkpoints/best.ckpt \
  --eval-file data/emoji_reply/heldout_validation.jsonl \
  --device cuda
```

Run OpenRouter frontier infill evaluation:

```bash
OPENROUTER_API_KEY=... python scripts/run_openrouter_infill_eval.py \
  --out-dir outputs/emoji_infill_full_eval_frontier/openrouter_gpt55_gemini_samples7 \
  --samples 7 \
  --workers 8 \
  --resume
```

## Repository Map

```text
configs/data/emoji_reply.yaml       emoji-reply dataset config
configs/data/text2emoji.yaml        text-to-emoji grounding config
configs/model/tiny-emoji.yaml       small local smoke-test model
data/emoji_reply/                   benchmark, heldout, and probe JSONL files
scripts/demo_emoji.py               terminal generation demo
scripts/serve_emoji.py              FastAPI browser demo
scripts/run_two_phase_h100.sh       two-phase training entrypoint
scripts/eval_permutation_stability.py
scripts/eval_infilling.py
scripts/run_openrouter_emoji_eval.py
scripts/run_openrouter_infill_eval.py
outputs/*/eval/                     committed evaluation artifacts
docs/decks/model_architecture.md    project architecture deck
```

## Evaluation Artifacts

Important committed outputs:

- `outputs/emoji_reply/eval/`
- `outputs/emoji_infill_full_eval_smoke_best/`
- `outputs/emoji_infill_full_eval_frontier/openrouter_gpt55_gemini_samples7/`
- `outputs/emoji_phase2_clean_promptbag_phase1_full_h100_1/eval/`
- `outputs/emoji_phase2_infill_promptbag_h100_long1/eval/`
- `outputs/emoji_phase2_infill_promptbag_h100_reg2/eval/`
- `outputs/emoji_phase2_infill_promptbag_h100_smoke/eval/`

The main W&B writeup is linked at the top of this README.

## Data

The reply dataset at `data/emoji_reply/emoji_reply.jsonl` (2,520 rows) is
**synthetic and template-generated** by `scripts/build_emoji_reply_dataset.py`.
Rows use an instruction-tuning shape:

```json
{"instruction": "I just got promoted", "input": "", "output": "🎉🥳👏", "topic": "celebration"}
```

How it is generated matters, so it is worth stating plainly. The script holds a
hand-authored `TOPICS` table mapping each topic to a pool of emoji, and then:

```python
for reply in emoji_combos(pool, k_min, k_max, per_message, rng):
    prompt = emoji_combos(pool, k_min, k_max, 1, rng)[0]
```

Both the prompt and the reply are independent random draws from the *same*
topic pool. See [Limitations](#limitations) for what that implies.

Additional evaluation sets:

- `data/emoji_reply/benchmark.jsonl`
- `data/emoji_reply/heldout_validation.jsonl`
- `data/emoji_reply/curated_permutation.jsonl`
- `data/emoji_reply/directional_probe.jsonl`

## Limitations

### The data generator assumes the hypothesis it was meant to test

Because prompt and reply are independent draws from one topic pool, **no
information flows from a prompt to its reply beyond the topic**, and neither
side carries order. Two things follow:

- **Order-invariance is a property of the generator, not a finding.** The
  targets are unordered bags by construction, so a model that ignores order is
  reproducing how the data was made. This cannot be evidence that emoji
  expressions are inherently order-free.
- **"Human fidelity" is not human.** The six references used for best-of-6
  scoring are six random draws from the same pool. Scoring well means having
  learned the pool. Our model trained on it; the frontier models never saw it.
  The reported 3–17× gap largely reflects that asymmetry rather than better
  emoji reasoning.

### RETRACTED: the order-effect result

**The order-effect finding in
`outputs/emoji_two_phase_h100_hard1/eval/ORDER_EFFECT_RESULTS.md` is withdrawn.**
Six independent problems were found, each verifiable in this repository. Any
one of the first four is disqualifying on its own.

**1. The order-invariance was trained in, not discovered.**
`outputs/emoji_two_phase_h100_hard1/phase2_reply/.hydra/config.yaml`:

```yaml
permutation_augment_prob: 1.0
permutation_augment_prompt: true
permutation_augment_response: true
```

Every Phase-2 example had its prompt *and* response randomly shuffled. Order
invariance was injected as data augmentation at probability 1.0. An
autoregressive model trained the same way would acquire the same invariance,
so the result says nothing about diffusion's inductive bias. Shuffling the
response additionally means response order *could not* be learned, and the
model is then scored with an order-blind metric. The pipeline assumes the
conclusion, enforces it, and measures it with a ruler that cannot see the
difference.

**2. It was evaluated on training data, with a checkpoint this repo already
labels invalid.** `eval_order_effect.py` defaults to `--split-partition all`,
which returns rows unchanged, selecting the first qualifying rows of
`data/emoji_reply/emoji_reply.jsonl` — the Phase-2 *training* file, including
all six reference replies per prompt. Meanwhile
`outputs/emoji_phase2_clean_promptbag_phase1_full_h100_1/eval/NO_LEAK_RESULTS.md`
names this checkpoint `leaky_hard1_last` and states it is "reference-only …
not valid … not admissible evidence." On a clean prompt-bag holdout the
comparable numbers are bag-Jaccard 0.060–0.106, against GPT-5.5's 0.101.

**3. One frontier baseline failed most of the time, and empty scored as
perfect.** `bag_jaccard("", "")` returns `1.0`. Gemini-3.1-Pro returned an
empty reply in 33/58 permutation samples (57%) and 37/58 resample samples
(64%), contaminating all 24 groups. Empty-matching-empty inflated its
`resample_stability`, and therefore its `order_effect`. Dropping GPT-5.5's
affected groups flips its `order_effect` from −0.022 to +0.022.

**4. The autoregressive baseline is structurally rigged, and was never run.**
`scripts/eval_permutation_stability.py:249` decodes AR with
`logits.argmax(dim=-1)` — deterministic. A deterministic model has
`resample_stability ≡ 1.0`, forcing `order_effect ≡ 1 − perm_stability`. It
cannot score well regardless of behaviour. No trained AR run exists anywhere
in `outputs/`, though `EMOJI_MODEL_PROCESS.md` calls it "the key comparison."

**5. Every effect is inside the noise.** Bootstrapping over the 24 groups, four
of five confidence intervals contain zero, and MDLM's two decoding
configurations produce *opposite signs* (−0.054 capped, +0.044 uncapped) with
intervals wider than the effect. No error bars were reported.

**6. The metric is not scale-free, and the checkpoint was chosen on it.**
`order_effect = resample − perm` is a raw difference on a bounded scale, so a
model with low `resample_stability` has little room to fall further and scores
near zero automatically — it rewards the sampler noise it was meant to
control for. Normalising to `1 − perm/resample` inverts the ranking:

| model | order_effect | scale-free |
|---|---:|---:|
| MDLM capped-3 | −0.054 | −0.096 |
| MDLM uncapped | +0.044 | **+0.152** |
| openai/gpt-5.5 | −0.022 | −0.039 |
| anthropic/claude-opus-4.8 | +0.178 | +0.339 |
| google/gemini-3.1-pro | +0.204 | +0.329 |

Under the scale-free form MDLM uncapped is *more* order-sensitive than
GPT-5.5 — the opposite of the published claim. Separately, `HARD_RESULTS.md`
records that validation NLL was best at step 200 but a later checkpoint was
used "because it showed better permutation stability," i.e. selection on the
outcome variable.

Independently, `eval/HELDOUT_ORDER_EFFECT_RESULTS.md` re-ran the frontier
baselines on a held-out split and the separation did not reproduce: Opus fell
0.178 → 0.069, Gemini flipped sign 0.204 → −0.044.

### The premise is weaker than stated, and diffusion does not follow from it

Even setting the measurements aside, the argument has two gaps.

**The linguistic premise is true only in a deflationary sense.** Cohn et al.
(2019), [*The grammar of emoji?*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6717234/),
find emoji "lack grammatical structure on their own" — but because emoji-only
utterances stay at "simplistic levels of patterning, primarily appearing as
one-unit utterances." Order-invariance over one or two symbols is trivial, not
evidence of a distinct holistic semantics. Herring & Ge
([2020](https://workshop-proceedings.icwsm.org/pdf/2020_05.pdf)) argue the
opposite case for emergent syntax. This repo's own
`data/emoji_reply/directional_probe.jsonl` collects counterexamples where order
plainly matters (`🐈🐁`, `🥚🐣🐔`, `🔴🟡🟢`) — though as written it cannot detect
order sensitivity, since every "chase" item maps to the same target.

**Order-invariance does not imply diffusion.** An AR model can represent an
exchangeable distribution given permutation augmentation, or by canonically
sorting the prompt — a one-line change. More fundamentally, permutation
stability measures invariance of the *prompt* encoding, while AR-vs-diffusion
is a choice about factorizing the *output*. And masked diffusion is provably
close to any-order autoregression
([arXiv:2511.19152](https://arxiv.org/abs/2511.19152)), so the real axis is
any-order vs fixed-order training, not diffusion vs AR.

### A defensible thesis is available

[*Diffusion Beats Autoregressive in Data-Constrained Settings*](https://arxiv.org/abs/2507.15857)
finds masked diffusion wins when data is scarce and compute is abundant,
because random masking acts as implicit augmentation. Phase 2 here was ~600
epochs over 2,520 examples — squarely that regime. That is the correct version
of the intuition, it is supported by published results, and the existing runs
already speak to it.

### What the project does support

- **Infilling.** The strongest result, and the only one measured on a clean
  prompt-bag holdout with benchmark rows excluded: power@k rises 0.2092 → 0.4295
  from k=1 to k=7, against GPT-5.5's 0.1879 → 0.2293 and Opus-4.8's 0.1834 →
  0.2241. **The slope is the finding** — diffusion samples diversely, so extra
  draws keep finding new correct fills while AR models converge on one answer.
  This is a diffusion property the literature predicts, cleanly measured.
  (Caveat: the local model is fine-tuned for infilling and the frontier models
  are zero-shot, so the slope is more defensible than the level.)
- **An atomic emoji tokenizer**, correctly scoped. Measured token counts:

  | encoding | 😀 | 🔥 | 👍🏽 | 🇺🇸 | 👨‍👩‍👧‍👦 |
  |---|---:|---:|---:|---:|---:|
  | gpt2 | 2 | 3 | 5 | 6 | 14 |
  | cl100k_base | 2 | 3 | 6 | 6 | 18 |
  | o200k_base | **1** | **1** | 3 | 4 | 11 |

  Modern tokenizers already handle *simple* emoji atomically, so the honest
  claim is narrower than "BPE fragments emoji": the win is on **compound**
  emoji — ZWJ sequences, flags, skin-tone modifiers — plus clean masking
  granularity, since each denoising step commits one symbol rather than one byte.
- **Making an unconditional MDLM conditional** by clamping a prefix and
  denoising only the response span, with no change to the training objective.
- **The resample-control instinct.** Separating order-sensitivity from sampler
  noise is the right idea. The *implementation* is flawed — see problem 6 above
  — but a scale-free variant with paired permutation tests and confidence
  intervals would be a real contribution.

### What would fix this

**Real data — now collected.** `data/emoji_reply/bluesky_pairs.jsonl` holds
18,563 emoji reply pairs from the public Bluesky firehose, where each reply is
a genuine human response to that specific prompt. It passes
`scripts/check_prompt_dependence.py` at +292.7 sd; the synthetic set fails at
−0.9 sd.

**The experiment that would actually test the thesis**, which has not been run:
a 2×2 over {diffusion, AR} × {`permutation_augment_prob` 0.0, 1.0}, with AR
decoded stochastically at matched temperature rather than `argmax`. Right now
architecture and augmentation are perfectly confounded. If augmented AR is as
order-invariant as augmented diffusion — the likely outcome — that is a clean
negative result worth reporting.

Also needed: prompts long enough to have order (4–6 emoji; 91% of the current
set is two), held-out evaluation rather than `--split-partition all`, empty
API responses filtered and reported as a first-class failure rate, and
checkpoint selection on validation loss rather than on the reported metric.

### Prior work this should cite

**EmojiLM** ([arXiv:2311.01751](https://arxiv.org/abs/2311.01751)) is the source
of the `KomeijiForce/Text2Emoji` dataset used in Phase 1, and it is a T5
seq2seq — an *autoregressive* model already doing text↔emoji generation. It is
the nearest prior work and the most meaningful head-to-head available, and it
is currently uncited.

Emoji prediction as classification is well established (DeepMoji, SemEval-2018
Task 2, emoji2vec, TweetEval). What appears genuinely unclaimed is an
emoji→emoji *conversational reply* model, and discrete diffusion applied to
emoji.

## Notes for Reviewers

This is not a wrapper around a text LLM. The final reply model uses an atomic
emoji vocabulary and generates emoji tokens directly with masked diffusion.
Text appears only as supervision in the semantic-grounding phase and in the
synthetic dataset construction/evaluation tooling.

The project was built in a day as a hackathon entry. Read it as an
infrastructure and evaluation artifact — tokenizer, conditional-sampling
method, and measurement harness — rather than as a validated claim about how
emoji semantics work.

## Credits

Built from the MDLM codebase:

```bibtex
@inproceedings{sahoo2024simple,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and Arriola, Marianne and Schiff, Yair and Gokaslan, Aaron and Marroquin, Edgar and Chiu, Justin T and Rush, Alexander and Kuleshov, Volodymyr},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
