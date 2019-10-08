import json
import pandas as pd
import tqdm
from lib.preprocess import TextPreprocess
from joblib import Parallel, delayed
import os
from collections import Counter

BOS = '_BOS_'
EOS = '_EOS_'
UNK = '_UNK_'


class CommentVocab(object):
    def __init__(self, load=True,
                 text_paths=['data/train.csv', 'data/test.csv'],
                 save_path='vocab/'):
        super(CommentVocab, self).__init__()
        self.text_preprocess = TextPreprocess()
        if load:
            print('Loading saved vocabulary')
            with open(save_path + 'saved_vocab.json', 'r') as v_file:
                vocab = json.load(v_file)
                self.tokens = vocab['tokens']
                self.token2id = vocab['tokens_to_id']
            print('Done')
        else:
            print("Generating vocabulary from texts")
            self.tokens = []
            for text_file in text_paths:
                df = pd.read_csv(text_file)
                data = Parallel(n_jobs=4)(delayed(self.text_preprocess)(comment)
                                          for comment in tqdm.tqdm(df['comment_text'], desc=f"Processing file: {text_file}"))
                for lst in data:
                    self.tokens.extend(lst)
            counter = Counter()
            counter.update(self.tokens)
            self.tokens = [word for word, count in counter.most_common(10000)]
            self.tokens = [BOS, EOS, UNK] + list(sorted(self.tokens))
            self.token2id = {token: idx for idx, token in enumerate(self.tokens)}
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + 'saved_vocab.json', 'w') as write_file:
                json.dump({'tokens': self.tokens, 'tokens_to_id': self.token2id}, write_file)
            print(f"Done. Vocab_size: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens)

    def text2ids(self, text):
        return [0] + [self.token2id[UNK] if token not in self.token2id else self.token2id[token]
                      for token in self.text_preprocess(text)]

    def words2ids(self, words):
        return [0] + [self.token2id[UNK] if word not in self.token2id else self.token2id[word]
                      for word in words]

    def ids2text(self, ids):
        return ' '.join(self.tokens[i] for i in ids)