"""Convenience runner: eval the emoji-diffusion model and one or more frontier
models on the same task, then print the headline leaderboard.

  python eval/run.py \
      --diffusion outputs/emoji_run/checkpoints/last.ckpt \
      --frontier openai/gpt-4o-mini anthropic/claude-3-5-haiku-latest \
      --epochs 8 --steps 128 --relevance-grader openai/gpt-4o-mini

Importing this module registers the custom provider, so model resolution works
without any CLI-timing surprises. For a no-checkpoint wiring test:
  python eval/run.py --frontier mockllm/model --epochs 2
"""
import argparse
import sys

import emoji_provider  # noqa: F401  (registers `emoji_diffusion`)
from inspect_ai import eval as inspect_eval
from emoji_task import emoji_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", help="path to MDLM emoji checkpoint")
    ap.add_argument("--frontier", nargs="*", default=[],
                    help="frontier model ids, e.g. openai/gpt-4o-mini")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--model-cfg", default="tiny-emoji")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--relevance-grader", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    models = list(args.frontier)
    model_args = {}
    if args.diffusion:
        models.insert(0, f"emoji_diffusion/{args.diffusion}")
        model_args = {"steps": args.steps, "model": args.model_cfg}
    if not models:
        sys.exit("provide --diffusion and/or --frontier models")

    logs = inspect_eval(
        emoji_task(relevance_grader=args.relevance_grader,
                   temperature=args.temperature),
        model=models,
        model_args=model_args,
        epochs=args.epochs,
        limit=args.limit,
    )

    print("\n=== headline: human_fidelity (higher = better) ===")
    rows = []
    for log in logs:
        name = log.eval.model
        hf = None
        if log.results:
            for s in log.results.scores:
                hf = s.metrics.get("human_fidelity")
                if hf is not None:
                    hf = hf.value
                    break
        rows.append((hf if hf is not None else -1, name))
    for hf, name in sorted(rows, reverse=True):
        print(f"  {hf:6.3f}  {name}")


if __name__ == "__main__":
    main()
