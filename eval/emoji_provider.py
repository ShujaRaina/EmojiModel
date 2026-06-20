"""Inspect AI model provider for the MDLM emoji-diffusion checkpoint.

Registers an `emoji_diffusion` provider so the model can be evaluated side by
side with frontier models in the same Inspect task:

    inspect eval eval/emoji_task.py \
        --model emoji_diffusion/outputs/emoji_run/checkpoints/last.ckpt \
        -M steps=128 -M model=tiny-emoji

The provider clamps the `<eot> text <eot>` prefix and lets the masked
diffusion model infill the emoji span -- the same path as
scripts/sample_emoji.py, refactored to load once and sample per request.
RNG is *not* fixed per call, so repeated generations (Inspect `epochs`) yield
the stochastic diversity that the order-symmetry / diversity metrics measure.
"""
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)


@modelapi(name="emoji_diffusion")
def emoji_diffusion():
    return EmojiDiffusionAPI


class EmojiDiffusionAPI(ModelAPI):
    def __init__(
        self,
        model_name,
        base_url=None,
        api_key=None,
        api_key_vars=None,
        config=GenerateConfig(),
        steps=128,
        length=64,
        model="tiny-emoji",
        device="cpu",
        **kwargs,
    ):
        super().__init__(model_name, base_url, api_key,
                         api_key_vars or [], config)
        # `model_name` is the checkpoint path passed after the slash.
        self.checkpoint = model_name
        self.steps = int(steps)
        self.length = int(length)
        self._load(model, device)

    def _load(self, model_cfg, device):
        import os
        import sys
        import hydra
        import torch

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # Lightning ckpts embed the omegaconf config; allow full unpickle.
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

        import dataloader
        import diffusion

        with hydra.initialize(version_base=None,
                              config_path="../configs"):
            cfg = hydra.compose(config_name="config", overrides=[
                "mode=sample_eval",
                "data=text2emoji",
                f"model={model_cfg}",
                f"model.length={self.length}",
                "parameterization=subs",
                "backbone=dit",
                f"trainer.accelerator={device}",
                "trainer.devices=1",
                f"sampling.steps={self.steps}",
                f"eval.checkpoint_path={self.checkpoint}",
            ])
        self.torch = torch
        self.tokenizer = dataloader.get_tokenizer(cfg)
        self.eot_id = self.tokenizer.eos_token_id
        self.model = diffusion.Diffusion.load_from_checkpoint(
            self.checkpoint, tokenizer=self.tokenizer, config=cfg)
        self.model.eval()

    def _prompt_text(self, input):
        for msg in reversed(input):
            if isinstance(msg, ChatMessageUser):
                return msg.text
        return input[-1].text if input else ""

    def _sample_one(self, text):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids[: self.length - 3]
        prefix = [self.eot_id] + ids + [self.eot_id]
        out = self.model.restore_model_and_cond_sample(
            [prefix], num_steps=self.steps)
        toks = out.cpu().tolist()[0][len(prefix):]
        emoji_ids = []
        for t in toks:
            if t == self.eot_id:
                break
            emoji_ids.append(t)
        # byte-level BPE can leave a dangling partial UTF-8 byte -> U+FFFD
        return self.tokenizer.decode(emoji_ids).replace("�", "").strip()

    async def generate(self, input, tools, tool_choice, config):
        text = self._prompt_text(input)
        with self.torch.no_grad():
            completion = self._sample_one(text)
        return ModelOutput.from_content(
            model=self.model_name, content=completion)

    def max_connections(self):
        # single in-process model; avoid concurrent forward passes
        return 1
