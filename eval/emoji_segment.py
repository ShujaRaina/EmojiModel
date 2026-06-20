"""Dependency-free emoji segmentation + classification.

Splits a string into grapheme clusters (handling ZWJ sequences like 👨‍🍳,
variation selectors 😮‍💨, skin-tone modifiers, keycaps 5️⃣, and regional-
indicator flag pairs) and classifies clusters as emoji vs. not. Uses the
`regex` module's \\X if available, else a hand-rolled clusterer.

This lets us compute, from a raw model completion:
  - extract_emoji(s): the ordered list of emoji (drops any text)
  - pure_emoji_ratio(s): fraction of visible clusters that are emoji
    (frontier models leak words/preamble; this catches it)
"""
ZWJ = 0x200D
VS16, VS15 = 0xFE0F, 0xFE0E
KEYCAP = 0x20E3
SKIN = range(0x1F3FB, 0x1F400)
REGIONAL = range(0x1F1E6, 0x1F200)

# Codepoint ranges that count as emoji for our purposes. Deliberately broad
# enough to cover the emoji that appear in the dataset (clocks ⏰, plus ➕,
# coffee ☕, star ⭐, arrows ⬆️, etc.) without sweeping in ordinary text.
_EMOJI_RANGES = [
    (0x231A, 0x231B), (0x2300, 0x23FF),  # watches, hourglass, ⏰⏳ tech symbols
    (0x25AA, 0x25FE),                    # small geometric (▪️▶️ etc.)
    (0x2600, 0x27BF),                    # misc symbols + dingbats (☀☕✨❤➕✅)
    (0x2934, 0x2935), (0x2B00, 0x2BFF),  # arrows, ⭐⬆️⬇️
    (0x1F000, 0x1FAFF),                  # main emoji blocks (incl. flags)
]


def _cp_is_emoji(cp):
    for lo, hi in _EMOJI_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def segment_graphemes(s):
    """Best-effort grapheme clustering."""
    try:
        import regex  # type: ignore
        return regex.findall(r"\X", s)
    except Exception:
        pass
    clusters, cur, prev = [], [], None
    for ch in s:
        cp = ord(ch)
        join = (
            cur
            and (
                cp in (ZWJ, VS16, VS15, KEYCAP)
                or cp in SKIN
                or prev == ZWJ
                # second half of a regional-indicator flag pair
                or (cp in REGIONAL and prev in REGIONAL and len(cur) == 1)
            )
        )
        if join:
            cur.append(ch)
        else:
            if cur:
                clusters.append("".join(cur))
            cur = [ch]
        prev = cp
    if cur:
        clusters.append("".join(cur))
    return clusters


def cluster_is_emoji(cluster):
    return any(
        _cp_is_emoji(ord(c)) or ord(c) in (KEYCAP, ZWJ) for c in cluster
    )


def extract_emoji(s):
    """Ordered list of emoji clusters in `s` (text is dropped)."""
    return [c for c in segment_graphemes(s) if cluster_is_emoji(c)]


def pure_emoji_ratio(s):
    """Fraction of visible (non-whitespace) clusters that are emoji.

    1.0 means the output is pure emoji; lower means the model leaked text,
    preamble, or punctuation.
    """
    visible = [c for c in segment_graphemes(s) if c.strip()]
    if not visible:
        return 0.0
    n_emoji = sum(cluster_is_emoji(c) for c in visible)
    return n_emoji / len(visible)


if __name__ == "__main__":
    for t in ["🎸🎧", "Here's a reply: 😂🔥", "👨‍🍳 cooking 5️⃣", "😮‍💨"]:
        print(repr(t), "->", extract_emoji(t),
              f"pure={pure_emoji_ratio(t):.2f}")
