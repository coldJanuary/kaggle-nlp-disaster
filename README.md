# Kaggle — NLP with Disaster Tweets

**Competition:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)  
**Result:** F1 = 0.836 on the public test set

## Approach

1. **Baseline** — TF-IDF + Logistic Regression (F1 = 0.783)
2. **BERT fine-tuning** — `bert-base-uncased` fine-tuned for 3 epochs with linear warmup scheduler
3. **Post-processing** — keyword-based rule layer on top of BERT predictions for edge cases

## Results

| Model | F1 (CV) | F1 (LB) |
|-------|---------|---------|
| TF-IDF + LogReg (baseline) | 0.778 | 0.783 |
| BERT fine-tuned | 0.841 | 0.836 |

## Structure

```
kaggle-nlp-disaster/
├── notebooks/
│   ├── 01_eda.ipynb            # text length, keyword analysis, class balance
│   ├── 02_baseline.ipynb       # TF-IDF + LogReg baseline
│   └── 03_bert_finetune.ipynb  # BERT training loop + evaluation
├── src/
│   ├── dataset.py              # PyTorch Dataset for tokenised tweets
│   └── train.py                # training loop with early stopping
├── requirements.txt
└── README.md
```

## Stack

`Python` `PyTorch` `HuggingFace Transformers` `BERT` `scikit-learn` `pandas` `NumPy`
