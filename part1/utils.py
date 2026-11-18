import hashlib
import random
import re


def example_transform(example):
    return {**example, "text": example["text"].lower()}


def custom_transform(example):
    """Simulate an all-caps review with occasional adjacent-letter typos."""
    text = example["text"]
    rng = random.Random(int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"))

    def typo(match):
        word = match.group()
        if len(word) >= 5 and rng.random() < 0.35:
            positions = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]
            if positions:
                i = rng.choice(positions)
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        return word.upper()

    return {**example, "text": re.sub(r"[A-Za-z]+", typo, text)}
