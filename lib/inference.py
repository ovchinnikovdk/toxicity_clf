from torch.utils.data import DataLoader
import pandas as pd
from lib.dataset import TextInferenceDataset
from lib.vocab import CommentVocab
import tqdm
import torch


class OneModelInference:
    def __init__(self, model, csv_path, vocab=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.vocab = vocab if vocab is not None else CommentVocab(load=True)

        self.data = DataLoader(TextInferenceDataset(pd.read_csv(csv_path), self.vocab, device=self.device),
                               batch_size=256,
                               shuffle=False)
        self.columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __call__(self, csv_save):
        self.model.eval()
        df = None
        for ids, texts in tqdm.tqdm(self.data):
            cur_df = pd.DataFrame()
            cur_df['id'] = ids
            preds = torch.nn.Sigmoid()(self.model(texts))
            for idx, column in enumerate(self.columns):
                cur_df[column] = preds[:, idx].cpu().detach().numpy()
            if df is not None:
                df = df.append(cur_df, ignore_index=True)
            else:
                df = cur_df
        df.to_csv(csv_save, index=False)
