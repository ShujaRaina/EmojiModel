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

### The frontier order-effect comparison did not reproduce

`eval/HELDOUT_ORDER_EFFECT_RESULTS.md` re-ran the frontier baselines on the
held-out split. Claude Opus 4.8 fell from 0.178 to 0.069 and Gemini 3.1 Pro
flipped sign from 0.204 to −0.044. All three frontier models look approximately
order-invariant there, so the separation reported on 24 groups does not hold at
a wider sample. Those runs contain **no MDLM row** — producing one needs the
Phase-2 checkpoint, which was never committed and is gone with the released
H100 instance.

### What the project does support

- An emoji-native masked diffusion model that generates emoji directly, with an
  atomic grapheme vocabulary rather than byte-pair fragments.
- A working method for making an unconditional MDLM conditional, by clamping a
  prefix and denoising only the response span — no change to the training
  objective.
- An evaluation harness with a genuine methodological contribution: the
  **resample control**. Raw permutation stability conflates order-sensitivity
  with diffusion sampling noise; `order_effect = resample_stability −
  perm_stability` separates them. That control is what revealed the first
  "weak result" to be sampler variance rather than order-sensitivity.
- Infilling results, which are the most defensible number here — filling masked
  positions is a real capability test, though still measured on synthetic data.

### What would fix this

Real emoji-reply pairs, where a reply genuinely responds to *that* prompt
rather than to its topic. The tokenizer, training pipeline, and eval harness
are all data-agnostic and would carry over unchanged. A useful acceptance test:
shuffle replies among prompts within a topic and re-score — if the metric barely
moves, the data carries no prompt→reply signal. The current dataset fails that
test by construction.

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
