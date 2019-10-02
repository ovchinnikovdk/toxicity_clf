from torch.utils.data import Dataset
from lib.vocab import CommentVocab
import pandas as pd
import torch


class TextDataset(Dataset):
    def __init__(self, csv_path=None, df=None, vocab=None, pad_size=256):
        if csv_path is None and df is None:
            raise ValueError
        self.pad_size = pad_size
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = CommentVocab(load=True)
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['comment_text'].iloc[idx]
        text = self.vocab.text2ids(text)
        ids = torch.zeros(self.pad_size)
        text = torch.tensor(text).long()
        ids[:min(len(text), self.pad_size)] = text[:min(len(text), self.pad_size)]
        target = self.data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].iloc[idx].values
        return (ids.long(), min(len(text), self.pad_size)), torch.tensor(target).float()
