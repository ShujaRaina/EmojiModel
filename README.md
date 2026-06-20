# Emoji Diffusion

Emoji Diffusion is a masked-diffusion language model that replies in emoji.
The project adapts MDLM to treat emoji sequences as holistic semantic objects
instead of left-to-right text strings.

The core demo: give the model an emoji prompt, generate an emoji-only reply,
then shuffle the prompt order and compare how stable the reply semantics are.

[W&B report: Emoji](https://wandb.ai/derektang-the-university-of-chicago/emoji-model/reports/Emoji--VmlldzoxNzI4OTc5NQ)

## Why This Exists

Emoji messages are not always ordinary token sequences. These prompts can carry
similar intent even when their order changes:

```text
😊😡😢
😢🔥😊
😡😊😢
```

Autoregressive models make order the main structure. Masked diffusion sees the
whole expression while denoising, which makes it a better fit for emoji
semantics, permutation stability, and infilling.

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

The project includes a synthetic emoji-reply instruction dataset at
`data/emoji_reply/emoji_reply.jsonl`. Rows use an instruction-tuning shape:

```json
{"instruction": "I just got promoted", "input": "", "output": "🎉🥳👏", "topic": "celebration"}
```

Additional evaluation sets:

- `data/emoji_reply/benchmark.jsonl`
- `data/emoji_reply/heldout_validation.jsonl`
- `data/emoji_reply/curated_permutation.jsonl`
- `data/emoji_reply/directional_probe.jsonl`

## Notes for Judges

This is not a wrapper around a text LLM. The final reply model uses an atomic
emoji vocabulary and generates emoji tokens directly with masked diffusion.
Text appears only as supervision in the semantic-grounding phase and in the
synthetic dataset construction/evaluation tooling.

The hackathon thesis is simple: emoji strings are compact semantic expressions,
and diffusion is a better modeling assumption than strict next-token decoding
when order should not dominate meaning.

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
