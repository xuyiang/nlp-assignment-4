import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import hashlib
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
