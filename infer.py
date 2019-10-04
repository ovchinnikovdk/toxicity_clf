from models.bi_lstm import BiLSTM
from lib.vocab import CommentVocab
from lib.inference import OneModelInference
import torch

pad_size = 256
vocab = CommentVocab(load=True)
path = 'data/test.csv'
model = BiLSTM(vocab_size=len(vocab), pad_size=pad_size)
model.load_state_dict(torch.load('logdir/checkpoints/best.pth')['model_state_dict'])
predictor = OneModelInference(model, path, vocab=vocab)

predictor('bisltm_submit.csv')