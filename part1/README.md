# DS-GA 1011: Homework 4 Part 1

> ### Please start early.

## Set up environment
We recommend you set up a conda environment for the packages used in this part of the homework.
```bash
conda create -n dsga1011-hw4-part1 python=3.11
conda activate dsga1011-hw4-part1
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk.
```bash
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
>>> exit()
```

## Assignment runs

Run from this directory with the starter defaults (batch size 8, 3 epochs, learning
rate 5e-5, fixed right padding and truncation to 512, FP32). Debug output goes to
a separate directory. Each output alternates a predicted class and its reference label.

```bash
python main.py --train --eval --debug_train
python main.py --eval_transformed --debug_transformation
python main.py --train --eval --eval_transformed
python main.py --train_augmented --eval --eval_transformed
python analyze_lengths.py
```

The length analysis reports results separately on reviews for which both versions
fit within the 510 content-token budget. For offline use, A4_BERT_MODEL can point
to an official local bert-base-cased checkpoint, A4_IMDB_DIR to the official
train.parquet and test.parquet, and A4_ACCURACY_METRIC to a local copy of Hugging
Face Evaluate's accuracy metric. Otherwise the standard Hugging Face loaders apply.
