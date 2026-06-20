"""Build a held-out emoji-reply test set for the Inspect eval.

Each unique text instruction in data/emoji_reply/emoji_reply.jsonl appears 6x
with *different* human emoji replies. We collapse those into one Sample whose
`target` is the list of human reference replies (the human distribution), split
deterministically by topic so train/test are balanced and leak-free.

Output: eval/data/emoji_test.jsonl, one record per held-out prompt:
  {"prompt": <text>, "topic": <str>, "references": [<emoji str>, ...]}
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "emoji_reply", "emoji_reply.jsonl")
OUT_DIR = os.path.join(HERE, "data")
TEST_EVERY = 4  # every 4th prompt within a topic -> test (~25%)


def main():
    refs = defaultdict(list)
    topic = {}
    for line in open(SRC, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        refs[d["instruction"]].append(d["output"])
        topic[d["instruction"]] = d["topic"]

    by_topic = defaultdict(list)
    for prompt in refs:
        by_topic[topic[prompt]].append(prompt)

    test = []
    for t in sorted(by_topic):
        for i, prompt in enumerate(sorted(by_topic[t])):
            if i % TEST_EVERY == 0:
                # de-dup references but keep order/multiplicity info as the
                # observed human distribution.
                test.append({
                    "prompt": prompt,
                    "topic": t,
                    "references": refs[prompt],
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "emoji_test.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(test)} held-out prompts across "
          f"{len(by_topic)} topics -> {out_path}")


if __name__ == "__main__":
    main()
