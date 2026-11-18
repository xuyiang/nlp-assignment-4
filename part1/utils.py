import hashlib
import random
import re


def example_transform(example):
    return {**example, "text": example["text"].lower()}


def custom_transform(example):
    """Introduce one adjacent interior-letter swap in selected words."""
    text = example["text"]
    rng = random.Random(int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"))

    def typo(match):
        word = match.group()
        if len(word) >= 4 and rng.random() < 0.50:
            positions = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]
            if positions:
                i = rng.choice(positions)
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        return word

    return {**example, "text": re.sub(r"[A-Za-z]+", typo, text)}
