"""Measure WordPiece lengths and accuracy on pairs that fit without truncation."""
import json
import os
from pathlib import Path
import statistics

from datasets import load_dataset
from transformers import AutoTokenizer
from utils import custom_transform


def paired_lengths():
    local = os.getenv("A4_IMDB_DIR")
    test = (load_dataset("parquet", data_files={"test": str(Path(local) / "test.parquet")}, split="test")
            if local else load_dataset("imdb", split="test"))
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("A4_BERT_MODEL", "bert-base-cased"))
    raw, transformed = [], []
    for offset in range(0,len(test),256):
        texts = test[offset:offset+256]["text"]
        edited = [custom_transform({"text": t})["text"] for t in texts]
        for batch, output in [(texts,raw),(edited,transformed)]:
            ids = tokenizer(batch, add_special_tokens=False, verbose=False)["input_ids"]
            output.extend(map(len,ids))
    return raw, transformed


def summarize(raw, changed, pred_dir="."):
    mask = [i for i,(a,b) in enumerate(zip(raw,changed)) if max(a,b)<=510]
    output = {"mean_wordpieces":[statistics.mean(raw),statistics.mean(changed)],
              "over_510":[sum(n>510 for n in x) for x in (raw,changed)],
              "paired_untruncated":len(mask),"n":len(raw)}
    for name in ("out_original","out_transformed","out_augmented_original","out_augmented_transformed"):
        path=Path(pred_dir)/(name+".txt")
        if path.exists():
            values=list(map(int,path.read_text().splitlines()))
            if len(values)!=2*len(raw):
                raise ValueError(f"Incomplete output file {path}")
            output[name]=sum(values[2*i]==values[2*i+1] for i in mask)
    return output


if __name__ == "__main__":
    print(json.dumps(summarize(*paired_lengths()),indent=2))
