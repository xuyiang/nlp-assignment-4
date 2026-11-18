# DS-GA 1011: Homework 4 Part 2

> ### Please start early.

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment. Note that Part 2 pins different package versions than Part 1, so use a separate environment from Part 1.

Example of virtual environment creation using conda:
```bash
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
python -m pip install -r requirements.txt
```

You can refer to the [HPC Tutorials](https://github.com/Athul-R/NYU-HPC-Tutorials) for more information on how to use the NYU HPC.

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```bash
python evaluate.py
  --predicted_sql results/t5_ft_experiment_dev.sql
  --predicted_records records/t5_ft_experiment_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `t5_ft_experiment_test.sql` (for extra credit `t5_ft_experiment_ec_test.sql`)

For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `t5_ft_experiment_test.pkl` (for extra credit `t5_ft_experiment_ec_test.pkl`)

⚠️ Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in `data/test.nl` in the order they appear.

## Reproduce the reported experiments

Run from this directory. The command with `--finetune` uses google-t5/t5-small.
The other command starts from randomly initialized weights with the same config.
Each run selects a checkpoint by development loss, compares greedy and beam-three
decoding on the development set, and generates the test outputs once with the
higher development record F1. The checkpoint includes the tokenizer, generation
configuration, and the reversible SQL merge rules learned on training targets.

```bash
python train_t5.py --finetune --max_n_epochs 24 --patience_epochs 5 --learning_rate 0.0005 --batch_size 24 --compare_beams 3
python train_t5.py --max_n_epochs 55 --patience_epochs 8 --learning_rate 0.001 --batch_size 24 --compare_beams 3
python analyze.py --stats --checkpoint checkpoints/ft_experiments/experiment/best
python analyze.py
python analyze.py --pred_sql results/t5_ft_experiment_ec_dev.sql --pred_records records/t5_ft_experiment_ec_dev.pkl
```

To use a locally downloaded copy of the base model, set `A4_T5_MODEL` to
its directory. To regenerate the selected fine-tuned model outputs, place the
saved checkpoint files under `checkpoints/ft_experiments/experiment/best` and
run `python train_t5.py --finetune --inference_only`. For the scratch model use
`checkpoints/scr_experiments/experiment/best` and omit `--finetune`.
Add `--dev_only` to skip test inference.
