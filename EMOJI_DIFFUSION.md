# Emoji Diffusion — Design & Approach

A text → emoji generator built on **MDLM** (Masked Diffusion Language Models,
[kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)) trained on the
[KomeijiForce/Text2Emoji](https://huggingface.co/datasets/KomeijiForce/Text2Emoji)
dataset.

## 1. End goal

Given a free-text prompt, generate a short, relevant sequence of emojis:

```
"I love sunny beaches and ice cream"  ->  🏖️🍦☀️😎
"Feeling sad and lonely today"        ->  😢💔🥀
"Time to study hard for my exams"     ->  📚✏️😰⏰
```

We do this with a **masked discrete diffusion** model (not autoregressive):
the model learns to denoise a fully-masked emoji span into emojis, conditioned
on the text. MDLM is a strong, simple diffusion-LM that we adapt for this
conditional task.

## 2. Why diffusion, and how we make it *conditional*

MDLM as released is an **unconditional** text diffusion model: it starts from a
sequence of `[MASK]` tokens and iteratively denoises them into text. To turn it
into a **conditional** `text → emoji` model without changing the training
objective, we use **infilling**:

1. **Data format.** Every training example becomes a single fixed-length
   sequence with the gpt2 end-of-text token (`<eot>`, id `50256`) acting as the
   start marker, the text/emoji boundary, and the terminator:

   ```
   <eot>  <text tokens>  <eot>  <emoji tokens>  <eot>  <pad…>
   ```

   Padding also uses `<eot>` but is zeroed out by the attention mask, so it
   never contributes to the loss.

2. **Training (conditional objective).** Rather than the vanilla MDLM objective
   over the whole sequence, we train a *proper* conditional model: a per-token
   `cond_mask` marks the `<eot> text <eot>` prefix as clean conditioning that is
   **never noised** and **never contributes to the loss**. So 100% of the
   learning signal goes to denoising the emoji span, conditioned on the clean
   text. (Without this, most of the gradient is wasted reconstructing the long
   English text region and the model barely learns text→emoji — empirically it
   emitted words, not emojis, even after thousands of steps.) This mirrors
   sampling exactly: text is clean/clamped, only the emoji span is diffused.
   Implemented via `q_xt(..., cond_mask)` and the loss mask in
   `diffusion._loss` / `_forward_pass_diffusion`.

3. **Sampling (the conditional trick).** At generation time we *clamp* the
   `<eot> text <eot>` prefix (mark it as already known / un-maskable) and run
   the reverse diffusion only over the remaining positions. MDLM's update rule
   keeps any non-masked token fixed (`copy_flag`), so the text is preserved
   while the emoji span is denoised from masks. We decode whatever appears
   between the 2nd and 3rd `<eot>`.

This is implemented in `diffusion.Diffusion._cond_sample` /
`restore_model_and_cond_sample` and driven by `scripts/sample_emoji.py`.

## 3. Tokenizer

We use the **gpt2 byte-level BPE** tokenizer (the repo default). Byte-level BPE
encodes *any* unicode — including multi-codepoint emoji and ZWJ sequences — as a
sequence of byte tokens, so no custom emoji vocabulary is required. Emojis
typically become 2–6 tokens each; we cap the emoji span at 32 tokens.

## 4. Model

| | Local CPU demo (`tiny-emoji`) | GPU target (`small`) |
|---|---|---|
| backbone | DiT (`dit`) | DiT (`dit`) |
| hidden size | 256 | 768 |
| blocks / heads | 4 / 4 | 12 / 12 |
| sequence length | 64 | 128 |
| params | ~30M (mostly embeddings) | ~170M |
| attention | `F.scaled_dot_product_attention` (CPU) | flash-attn (GPU) |

The repo's DiT hard-depended on `flash-attn` (CUDA-only). We added a portable
fallback so the **exact same code runs on CPU and GPU** — flash-attn is used
when available, otherwise PyTorch SDPA. Same for the `mamba`/`AR` backbones'
imports (made optional).

## 5. Repository / where things run

The project is **one codebase, two execution modes** — you only change a few
flags, not the code.

```
mdlm/
├── dataloader.py            # + text2emoji formatting (<eot> text <eot> emoji <eot>)
├── diffusion.py             # + conditional infilling sampler
├── models/dit.py            # + CPU / no-flash-attn attention fallback
├── configs/
│   ├── data/text2emoji.yaml
│   ├── data/emoji_reply.yaml         # local instruction-tuning set
│   ├── model/tiny-emoji.yaml
│   └── strategy/single_device.yaml   # CPU single-process
├── data/
│   └── emoji_reply/emoji_reply.jsonl # generated emoji-reply dataset
├── scripts/
│   ├── train_text2emoji.sh           # GPU training
│   ├── train_text2emoji_cpu.sh       # CPU training (this VM)
│   ├── build_emoji_reply_dataset.py  # generates the emoji-reply dataset
│   └── sample_emoji.py               # conditional text -> emoji generation
└── requirements-cpu.txt              # CPU install (no flash-attn/mamba)
```

### What runs locally (this Devin CPU VM)
- **Development & integration**: all code, configs, glue.
- **Data prep**: download + tokenize the dataset once (cached to
  `~/mdlm_data`). This is CPU-bound and identical to what the GPU run uses.
- **Proof-of-concept training**: the `tiny-emoji` model for a few thousand
  steps (~1.5 s/step on 8 CPU cores) to prove the full loop end-to-end.
- **Sampling / smoke tests**: generate emojis from any checkpoint.

The VM has **no GPU**, so local training is only good enough to validate the
pipeline and produce a weak demo model — not a high-quality one.

### What should run on the cloud (GPU)
- **Real training**: `model=small`, longer sequences, larger batch, 30k–100k
  steps. A single mid-range GPU (e.g. A5000/3090/A100) trains this in a few
  hours. Command: `bash scripts/train_text2emoji.sh`.
- **Large-batch sampling / evaluation**.

The dataset cache and code are portable; moving to GPU is just `git pull`,
install the GPU deps (`requirements.yaml`), and run the GPU script.

## 5b. The emoji-reply instruction-tuning dataset

Alongside the large `Text2Emoji` corpus, the repo ships a small synthetic
**emoji-reply** dataset for instruction tuning an emoji-style model:
`data/emoji_reply/emoji_reply.jsonl` (~2.5k rows), generated by
`scripts/build_emoji_reply_dataset.py`.

Each row is a natural message paired with an emoji-only reply, in Alpaca-style
instruction-tuning format:

```json
{"instruction": "I just got promoted", "input": "", "output": "🎉🥳👏", "topic": "celebration"}
```

It is built from ~28 topics (love, celebration, food, sad, gym, weather,
study, travel, animals, music, coffee, anger, …); for each message we sample
several distinct on-topic emoji combos (seeded → reproducible) and dedupe.
Regenerate / resize it with:

```bash
python scripts/build_emoji_reply_dataset.py --per-message 8 --seed 42
```

It is directly trainable by the same conditional pipeline — `instruction` maps
to the text prefix, `output` to the emoji span:

```bash
python main.py mode=train data=emoji_reply model=tiny-emoji \
  model.length=64 backbone=dit strategy=single_device \
  trainer.accelerator=cpu trainer.devices=1
```

## 6. Cloud compute — recommendation

Local CPU is fine for the demo, but for a model that produces genuinely good
emojis we want a GPU. Options, roughly in order of convenience:

1. **Lightning Studio** — the MDLM repo already ships a Studio badge; least
   friction, environment pre-baked.
2. **RunPod / Vast.ai / Lambda** — cheap on-demand single-GPU boxes
   (A5000/3090 ≈ \$0.2–0.5/hr; a full run is a few \$).
3. **Google Colab (Pro)** — quickest to try, but flash-attn install can be
   fiddly; our SDPA fallback means it'll still run without it.
4. **Your own GPU / cluster** — the repo also has Slurm scripts.

I can target whichever you pick and kick off the run. My recommendation: a
single A5000/A100 on RunPod or Lambda for one solid training run.

## 7. Weights & Biases — recommendation

**Yes, recommended for the real run.** MDLM logs train/val NLL, perplexity,
learning rate, and sample generations to wandb out of the box.

- **Locally** we run with `WANDB_MODE=disabled` (no account needed) since the
  demo is throwaway.
- **On the cloud run** we should enable it: set `WANDB_API_KEY`, drop the
  `disabled` flag, and set `wandb.name=...`. You'll get live loss curves and
  logged emoji samples per validation step.

If you'd rather not use wandb, the runs still work with it disabled — we'd just
watch the console / TensorBoard-style logs instead.

## 8. Status

- Pipeline implemented and validated end-to-end on CPU (data → train →
  conditional sampling all run; outputs are correctly structured).
- Conditional objective (`cond_mask`) in place — the `tiny-emoji` CPU model
  produces real emojis after ~1k steps (relevance improves with more
  training / a GPU `model=small` run).
- Synthetic emoji-reply instruction-tuning dataset generated and wired in.
- GPU training for a quality model — available via `scripts/train_text2emoji.sh`.
- Shipped as PR on `ShujaRaina/EmojiModel`.
