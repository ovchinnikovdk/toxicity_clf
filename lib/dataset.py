from torch.utils.data import Dataset
from lib.vocab import CommentVocab
import pandas as pd
import torch


class TextTrainDataset(Dataset):
    def __init__(self, df, vocab, pad_size=256):
        super(TextTrainDataset, self).__init__()
        self.pad_size = pad_size
        self.vocab = vocab
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


class TextInferenceDataset(Dataset):
    def __init__(self, df, vocab, pad_size=256, device='cpu'):
        super(TextInferenceDataset, self).__init__()
        self.pad_size = pad_size
        self.df = df
        self.vocab = vocab
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.df['id'].iloc[idx]
        text = self.df['comment_text'].iloc[idx]
        text = self.vocab.text2ids(text)
        text_tensor = torch.zeros(self.pad_size)
        text = torch.tensor(text).long()
        text_tensor[:min(len(text), self.pad_size)] = text[:min(len(text), self.pad_size)]
        return id, (text_tensor.long().to(self.device), min(len(text), self.pad_size))
