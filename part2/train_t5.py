import argparse
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from load_data import PAD_IDX, get_codec, load_t5_data
from t5_utils import (DEVICE, checkpoint_dir, initialize_model, initialize_optimizer_and_scheduler,
                      load_model_from_checkpoint, save_model, setup_wandb)
from utils import compute_metrics, save_queries_and_records, set_random_seeds


def get_args():
    parser = argparse.ArgumentParser(description="T5 text-to-SQL")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--optimizer_type", choices=["AdamW"], default="AdamW")
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--scheduler_type", choices=["none", "cosine", "linear"], default="cosine")
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=20)
    parser.add_argument("--patience_epochs", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", default="experiment")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--merges", type=int, choices=range(101), default=100, metavar="0..100")
    parser.add_argument("--num_beams", type=int, help="Default: checkpoint setting, or greedy decoding")
    parser.add_argument("--compare_beams", type=int, help="Also score this beam width on dev; choose by record F1")
    parser.add_argument("--max_new_tokens", type=int, default=224)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--dev_only", action="store_true")
    return parser.parse_args()


def mixed_precision():
    return torch.autocast(DEVICE.type, dtype=torch.bfloat16,
                          enabled=DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())


def batch_loss(model, batch):
    inputs, mask, decoder, targets = [x.to(DEVICE) for x in batch[:4]]
    with mixed_precision():
        logits = model(input_ids=inputs, attention_mask=mask, decoder_input_ids=decoder, use_cache=False).logits
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_IDX)
    return loss, targets.ne(PAD_IDX).sum().item()


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total = count = 0
    for batch in tqdm(train_loader, desc="Train", mininterval=30):
        optimizer.zero_grad(set_to_none=True)
        loss, n = batch_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * n
        count += n
    return total / count


@torch.inference_mode()
def validation_loss(model, loader):
    model.eval()
    total = count = 0
    for batch in loader:
        loss, n = batch_loss(model, batch)
        total += loss.item() * n
        count += n
    return total / count


@torch.inference_mode()
def generate_queries(args, model, loader):
    model.eval()
    codec = loader.dataset.codec
    queries = []
    for batch in tqdm(loader, desc="Generate", mininterval=30):
        inputs, mask, start = (batch[i].to(DEVICE) for i in (0, 1, -1))
        with mixed_precision():
            ids = model.generate(input_ids=inputs, attention_mask=mask, decoder_input_ids=start,
                                 max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, do_sample=False)
        queries.extend(codec.decode(row) for row in ids.tolist())
    return queries


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    loss = validation_loss(model, dev_loader)
    queries = generate_queries(args, model, dev_loader)
    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, f1, errors = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    return loss, f1, record_em, sql_em, sum(bool(e) for e in errors) / len(errors)


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    save_queries_and_records(generate_queries(args, model, test_loader), model_sql_path, model_record_path)


def train(args, model, train_loader, dev_loader, optimizer, scheduler, tracker=None):
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, args.max_n_epochs + 1):
        train_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        dev_loss = validation_loss(model, dev_loader)
        print(f"Epoch {epoch} train loss {train_loss:.5f} dev loss {dev_loss:.5f}", flush=True)
        if tracker:
            tracker.log({"train/loss": train_loss, "dev/loss": dev_loss}, step=epoch)
        if dev_loss < best_loss:
            save_model(checkpoint_dir(args), model, True, train_loader.dataset.codec)
            best_loss, stale = dev_loss, 0
        else:
            stale += 1
            if stale >= args.patience_epochs:
                break


def main():
    args = get_args()
    set_random_seeds(42)
    tracker = setup_wandb(args) if args.use_wandb else None
    checkpoint = checkpoint_dir(args) / "best"
    codec = get_codec("data", args.merges, str(checkpoint) if args.inference_only else None)
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size, codec)
    if not args.inference_only:
        model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler, tracker)
        del model, optimizer, scheduler
    model = load_model_from_checkpoint(args, best=True)
    name = f"t5_ft_{args.experiment_name}" + ("" if args.finetune else "_ec")
    Path("results").mkdir(exist_ok=True)
    Path("records").mkdir(exist_ok=True)
    widths = [args.num_beams or model.generation_config.num_beams or 1]
    if args.compare_beams and args.compare_beams not in widths:
        widths.append(args.compare_beams)
    best_f1 = -1
    for width in widths:
        args.num_beams = width
        tag = f"{name}_beam{width}_dev" if len(widths) > 1 else f"{name}_dev"
        sql_path, record_path = Path(f"results/{tag}.sql"), Path(f"records/{tag}.pkl")
        metrics = eval_epoch(args, model, dev_loader, "data/dev.sql", sql_path,
                             "records/ground_truth_dev.pkl", record_path)
        print(f"Dev beam={width}: loss / record F1 / record EM / SQL EM / SQL error rate: {metrics}", flush=True)
        if metrics[1] > best_f1:
            best_f1, best_width = metrics[1], width
            if len(widths) > 1:
                shutil.copyfile(sql_path, f"results/{name}_dev.sql")
                shutil.copyfile(record_path, f"records/{name}_dev.pkl")
    args.num_beams = best_width
    model.generation_config.num_beams = best_width
    model.generation_config.max_new_tokens = args.max_new_tokens
    model.generation_config.save_pretrained(checkpoint)
    print(f"Selected beam width {best_width}", flush=True)
    if not args.dev_only:
        test_inference(args, model, test_loader, f"results/{name}_test.sql", f"records/{name}_test.pkl")
    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()
