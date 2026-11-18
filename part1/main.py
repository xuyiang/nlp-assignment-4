import argparse
import os
import random
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler

from utils import custom_transform


def make_loader(args, dataset, shuffle=False):
    def encode(batch):
        # Retain both the setup and the final verdict in long reviews.
        tokens = args.tokenizer(batch["text"], add_special_tokens=False, verbose=False)["input_ids"]
        limit = args.max_length - 2
        head = limit // 2
        ids = [row if len(row) <= limit else row[:head] + row[-(limit - head):] for row in tokens]
        return {"input_ids": [args.tokenizer.build_inputs_with_special_tokens(row) for row in ids]}

    encoded = dataset.map(encode, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    collate = DataCollatorWithPadding(args.tokenizer, pad_to_multiple_of=8)
    return DataLoader(encoded, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)


def do_train(args, model, train_dataloader, save_dir="./out"):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    steps = len(train_dataloader) * args.num_epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=steps)
    model.train()
    for epoch in range(args.num_epochs):
        loss_sum = 0.0
        for batch in tqdm(train_dataloader, desc=f"Train {epoch + 1}", mininterval=30):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach().item()
        print(f"Epoch {epoch + 1}: mean batch loss {loss_sum / len(train_dataloader):.4f}", flush=True)
    model.save_pretrained(save_dir)
    args.tokenizer.save_pretrained(save_dir)


@torch.inference_mode()
def do_eval(eval_dataloader, output_dir, out_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device).eval()
    correct = total = 0
    with open(out_file, "w", encoding="utf-8") as out:
        for batch in tqdm(eval_dataloader, desc="Evaluate", mininterval=30):
            labels = batch.pop("labels").tolist()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                preds = model(**{k: v.to(device) for k, v in batch.items()}).logits.argmax(-1).tolist()
            for pred, label in zip(preds, labels):
                out.write(f"{pred}\n{label}\n")
                correct += pred == label
                total += 1
    return {"accuracy": correct / total, "correct": correct, "total": total}


def create_augmented_dataloader(args, dataset):
    original = dataset["train"]
    subset = original.shuffle(seed=42).select(range(min(5000, len(original))))
    transformed = subset.map(custom_transform)
    return make_loader(args, concatenate_datasets([original, transformed]), shuffle=True)


def create_transformed_dataloader(args, dataset, debug_transformation=False):
    split = dataset["test"]
    if debug_transformation:
        for ex in split.shuffle(seed=42).select(range(min(5, len(split)))):
            print("Original:", ex["text"], "\nTransformed:", custom_transform(ex)["text"])
        return None
    return make_loader(args, split.map(custom_transform))


def main():
    parser = argparse.ArgumentParser()
    for flag in ["train", "train_augmented", "eval", "eval_transformed", "debug_train", "debug_transformation"]:
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--model_dir", default="./out")
    parser.add_argument("--base_model", default=os.getenv("A4_BERT_MODEL", "bert-base-cased"))
    parser.add_argument("--imdb_dir", default=os.getenv("A4_IMDB_DIR"), help="Local official train/test Parquet files")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512, choices=range(8, 513), metavar="8..512")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.imdb_dir:
        dataset = load_dataset("parquet", data_files={s: str(Path(args.imdb_dir) / f"{s}.parquet") for s in ("train", "test")})
    else:
        dataset = load_dataset("imdb")
    if args.debug_transformation:
        create_transformed_dataloader(args, dataset, True)
        return
    args.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.debug_train:
        for split, size in [("train", 4000), ("test", 1000)]:
            dataset[split] = dataset[split].shuffle(seed=42).select(range(size))
    for enabled, augmented in [(args.train, False), (args.train_augmented, True)]:
        if enabled:
            train_loader = create_augmented_dataloader(args, dataset) if augmented else make_loader(args, dataset["train"], True)
            model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2).to(device)
            directory = "out_augmented" if augmented else "out"
            args.model_dir = directory + ("_debug" if args.debug_train else "")
            do_train(args, model, train_loader, args.model_dir)
            del model
    for enabled, transformed in [(args.eval, False), (args.eval_transformed, True)]:
        if enabled:
            loader = create_transformed_dataloader(args, dataset) if transformed else make_loader(args, dataset["test"])
            suffix = "transformed" if transformed else "original"
            output = f"{Path(args.model_dir).name}_{suffix}.txt"
            print(output, do_eval(loader, args.model_dir, output), flush=True)


if __name__ == "__main__":
    main()
