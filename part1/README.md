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

Run these commands from this directory. Each text output alternates a prediction
and its reference label, in dataset order. The debug run uses a separate output
directory. The full experiments use 320 tokens including special tokens and
keep both ends of longer reviews.

```bash
python main.py --train --eval --debug_train --max_length 320 --learning_rate 2e-5
python main.py --debug_transformation
python main.py --train --eval --eval_transformed --max_length 320 --learning_rate 2e-5
python main.py --train_augmented --eval --eval_transformed --max_length 320 --learning_rate 2e-5
```

Set `A4_BERT_MODEL` to a local bert-base-cased checkpoint and `A4_IMDB_DIR`
to a directory with the official train.parquet and test.parquet splits to run
from local copies. Without these options the loader uses Hugging Face.
