import torch
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    """PyTorch Dataset for tokenised disaster tweets."""

    def __init__(self, texts, tokenizer, labels=None, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
