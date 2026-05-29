import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.dataset import TweetDataset

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
MAX_LEN = 128
EPOCHS = 3
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)


def eval_epoch(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(pred)
            if "labels" in batch:
                targets.extend(batch["labels"].numpy())
    return np.array(preds), np.array(targets)


def train_cv(texts, labels, n_splits=5):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(labels))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"Fold {fold + 1}/{n_splits}")
        tr_ds = TweetDataset([texts[i] for i in tr_idx], tokenizer, [labels[i] for i in tr_idx], MAX_LEN)
        val_ds = TweetDataset([texts[i] for i in val_idx], tokenizer, [labels[i] for i in val_idx], MAX_LEN)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        total_steps = len(tr_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

        for epoch in range(EPOCHS):
            loss = train_epoch(model, tr_loader, optimizer, scheduler)
            preds, targets = eval_epoch(model, val_loader)
            f1 = f1_score(targets, preds)
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}  val_F1={f1:.4f}")

        preds, _ = eval_epoch(model, val_loader)
        oof_preds[val_idx] = preds

    print(f"OOF F1: {f1_score(labels, oof_preds):.4f}")
    return oof_preds
