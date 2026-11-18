from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast

PAD_IDX = 0
PROMPT = "translate to SQL: "


def load_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_prompting_data(data_folder):
    folder = Path(data_folder)
    return tuple(load_lines(folder / name) for name in ("train.nl", "train.sql", "dev.nl", "dev.sql", "test.nl"))


class SQLCodec:
    """Encode frequent adjacent SQL subwords with T5's existing sentinel IDs."""

    def __init__(self, tokenizer, rules):
        self.tokenizer = tokenizer
        self.rules = rules
        self.expansions = {}
        for left, right, sentinel in rules:
            self.expansions[sentinel] = self.expansions.get(left, [left]) + self.expansions.get(right, [right])

    @staticmethod
    def base_ids(tokenizer, queries):
        # "lessthan" is absent from the SQL corpus. T5 otherwise maps '<' to <unk>.
        return tokenizer([q.replace("<", "lessthan") for q in queries], add_special_tokens=False, verbose=False).input_ids

    @classmethod
    def fit(cls, tokenizer, queries, merges=100):
        if any("lessthan" in q for q in queries):
            raise ValueError("SQL escape collides with a training literal")
        rows = cls.base_ids(tokenizer, queries)
        # Each character represents one token, enabling lossless, non-overlapping pair replacement.
        rows = ["".join(map(chr, ids)) for ids in rows]
        rules = []
        for i in range(merges):
            counts = Counter(pair for row in rows for pair in zip(row, row[1:]))
            if not counts:
                break
            pair = counts.most_common(1)[0][0]
            sentinel = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            rules.append([ord(pair[0]), ord(pair[1]), sentinel])
            rows = [row.replace("".join(pair), chr(sentinel)) for row in rows]
        return cls(tokenizer, rules)

    def encode(self, queries):
        texts = ["".join(map(chr, row)) for row in self.base_ids(self.tokenizer, queries)]
        for left, right, sentinel in self.rules:
            pair = chr(left) + chr(right)
            texts = [row.replace(pair, chr(sentinel)) for row in texts]
        return [[*map(ord, row), self.tokenizer.eos_token_id] for row in texts]

    def decode(self, row):
        ids = []
        for token in row:
            if token == self.tokenizer.eos_token_id:
                break
            if token != PAD_IDX:
                ids.extend(self.expansions.get(token, [token]))
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False).replace("lessthan", "<").strip()

    def save(self, path):
        Path(path).write_text(json.dumps(self.rules), encoding="utf-8")


@lru_cache(maxsize=2)
def get_codec(data_folder="data", merges=100, checkpoint=None):
    model_name = checkpoint or os.getenv("A4_T5_MODEL", "google-t5/t5-small")
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    if checkpoint:
        rules = json.loads((Path(checkpoint) / "sql_codec.json").read_text(encoding="utf-8"))
        return SQLCodec(tokenizer, rules)
    return SQLCodec.fit(tokenizer, load_lines(Path(data_folder) / "train.sql"), merges)


class T5Dataset(Dataset):
    def __init__(self, data_folder, split, codec=None):
        self.codec = codec or get_codec(str(data_folder))
        self.examples = self.process_data(data_folder, split, self.codec.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        folder = Path(data_folder)
        prompts = [PROMPT + text for text in load_lines(folder / f"{split}.nl")]
        inputs = tokenizer(prompts, verbose=False).input_ids
        if split == "test":
            return [(torch.tensor(ids),) for ids in inputs]
        queries = load_lines(folder / f"{split}.sql")
        if len(inputs) != len(queries):
            raise ValueError(f"Unpaired {split} examples")
        return [(torch.tensor(x), torch.tensor(y)) for x, y in zip(inputs, self.codec.encode(queries))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def test_collate_fn(batch):
    inputs = pad_sequence([ex[0] for ex in batch], batch_first=True, padding_value=PAD_IDX)
    start = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return inputs, inputs.ne(PAD_IDX).long(), start


def normal_collate_fn(batch):
    inputs, mask, start = test_collate_fn(batch)
    targets = pad_sequence([ex[1] for ex in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.cat([start, targets[:, :-1]], dim=1)
    return inputs, mask, decoder_inputs, targets, start


class LengthGroupedBatches:
    """Shuffle examples each epoch, then group by target length in local windows."""
    def __init__(self, dataset, batch_size):
        self.lengths = [len(row[1]) for row in dataset.examples]
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.lengths)).tolist()
        window = 8 * self.batch_size
        batches = []
        for offset in range(0, len(order), window):
            group = sorted(order[offset:offset + window], key=self.lengths.__getitem__)
            batches.extend(group[i:i + self.batch_size] for i in range(0, len(group), self.batch_size))
        for idx in torch.randperm(len(batches)).tolist():
            yield batches[idx]


def get_dataloader(batch_size, split, codec=None):
    dataset = T5Dataset("data", split, codec)
    if split == "train":
        return DataLoader(dataset, batch_sampler=LengthGroupedBatches(dataset, batch_size), collate_fn=normal_collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=test_collate_fn if split == "test" else normal_collate_fn)


def load_t5_data(batch_size, test_batch_size, codec=None):
    return tuple(get_dataloader(batch_size if split == "train" else test_batch_size, split, codec)
                 for split in ("train", "dev", "test"))
