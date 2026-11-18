import os
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import T5Config, T5ForConditionalGeneration, get_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_wandb(args):
    import wandb
    return wandb.init(project="dsga1011-a4", name=args.experiment_name, config=vars(args))


def initialize_model(args):
    name = os.getenv("A4_T5_MODEL", "google-t5/t5-small")
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(name)
    else:
        model = T5ForConditionalGeneration(T5Config.from_pretrained(name))
    return model.to(DEVICE)


def checkpoint_dir(args):
    mode = "ft" if args.finetune else "scr"
    return Path("checkpoints") / f"{mode}_experiments" / args.experiment_name


def save_model(directory, model, best, codec=None):
    destination = Path(directory) / ("best" if best else "last")
    model.save_pretrained(destination)
    if codec is not None:
        codec.save(destination / "sql_codec.json")
        codec.tokenizer.save_pretrained(destination)


def load_model_from_checkpoint(args, best):
    path = checkpoint_dir(args) / ("best" if best else "last")
    return T5ForConditionalGeneration.from_pretrained(path).to(DEVICE)


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    optimizer = AdamW([{"params": decay, "weight_decay": args.weight_decay},
                       {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)
    schedule = None
    if args.scheduler_type != "none":
        schedule = get_scheduler(args.scheduler_type, optimizer, args.num_warmup_epochs * epoch_length,
                                 args.max_n_epochs * epoch_length)
    return optimizer, schedule
