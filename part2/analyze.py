"""Reproduce the Q4 tables or inspect Q6 execution and grounding errors."""
import argparse
from collections import Counter
from pathlib import Path
import re
from statistics import mean

from load_data import PROMPT, get_codec, load_lines
from utils import compute_metrics, load_queries_and_records


def data_statistics(checkpoint):
    codec = get_codec(checkpoint=checkpoint)
    for split in ("train", "dev"):
        questions = load_lines(Path("data") / f"{split}.nl")
        queries = load_lines(Path("data") / f"{split}.sql")
        encode = lambda strings: codec.tokenizer(strings, verbose=False).input_ids
        after_sql = codec.encode(queries, questions)
        exact = sum(codec.decode(ids, q) == sql for ids, q, sql in zip(after_sql, questions, queries))
        print(f"Target round trips: {exact}/{len(questions)}")
        make_prompt = getattr(codec, "input_text", lambda text: PROMPT + text)
        print(f"{split} ({len(questions)} examples)")
        for label, rows in (("raw NL", encode(questions)), ("raw SQL", encode(queries)),
                            ("processed NL", encode([make_prompt(s) for s in questions])),
                            ("processed SQL", after_sql)):
            print(f"  {label:14s} mean tokens {mean(map(len, rows)):.2f}, "
                  f"observed vocabulary {len({token for row in rows for token in row})}")


def projection(sql):
    select = re.match(r"SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM\b", sql, flags=re.I)
    return select.group(1).strip().lower() if select else None


def quoted_values(sql):
    return Counter(re.findall(r"'([^']*)'", sql))


def analyze_predictions(pred_sql, pred_records):
    questions = load_lines(Path("data/dev.nl"))
    gold, references, _ = load_queries_and_records("data/dev.sql", "records/ground_truth_dev.pkl")
    pred, records, errors = load_queries_and_records(pred_sql, pred_records)
    sql_em, record_em, f1, _ = compute_metrics("data/dev.sql", pred_sql, "records/ground_truth_dev.pkl", pred_records)
    print(f"SQL EM {sql_em:.5f}, record EM {record_em:.5f}, record F1 {f1:.5f}")
    categories = {k: [] for k in ("Execution error", "Different projection", "Different quoted values",
                                     "Different numeric constraints", "Other executable mismatch")}
    for i, (ref, hyp, expected, actual, error) in enumerate(zip(gold, pred, references, records, errors)):
        if error:
            key = "Execution error"
        elif set(expected) == set(actual):
            continue
        elif projection(ref) != projection(hyp):
            key = "Different projection"
        elif quoted_values(ref) != quoted_values(hyp):
            key = "Different quoted values"
        elif re.findall(r"[<>=]+\s*\d+", ref) != re.findall(r"[<>=]+\s*\d+", hyp):
            key = "Different numeric constraints"
        else:
            key = "Other executable mismatch"
        categories[key].append(i)
    for key, indices in categories.items():
        print(f"\n{key}: {len(indices)}/{len(gold)}")
        if indices:
            i = indices[0]
            print(f"  Dev row {i + 1}: {questions[i]}\n  Gold: {gold[i]}\n  Model: {pred[i]}\n  SQLite: {errors[i]}")
    return categories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--checkpoint", help="Read the tokenizer and SQL merge rules from this best checkpoint")
    parser.add_argument("--pred_sql", default="results/t5_ft_experiment_dev.sql")
    parser.add_argument("--pred_records", default="records/t5_ft_experiment_dev.pkl")
    args = parser.parse_args()
    if args.stats:
        data_statistics(args.checkpoint)
    else:
        analyze_predictions(args.pred_sql, args.pred_records)
