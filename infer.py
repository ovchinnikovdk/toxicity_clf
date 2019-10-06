from models.bi_lstm import BiLSTM
from models.bi_lstm_attn import BiLSTM_Attn
from models.cnn import TextCNN
from lib.vocab import CommentVocab
from lib.inference import OneModelInference
import torch
import argparse


vocab = CommentVocab(load=True)
pad_size = 256

models = {
    'bisltm': BiLSTM,
    'bilstm_attn': BiLSTM_Attn,
    'text_cnn': TextCNN

}
params = {
    'bilstm': {
        'vocab_size': len(vocab),
        'pad_size': pad_size
    },
    'bilstm_attn': {
        'vocab_size': len(vocab),
        'pad_size': pad_size
    },
    'text_cnn': {
        'vocab_size': len(vocab),
        'pad_size': pad_size
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help='Select model to train, for all (stacking) use \'all\'')
    parser.add_argument("--path", type=str, default="data/test.csv", help="Enter path to CSV-file")
    args = parser.parse_args()
    path = args.path
    model = models[args.model](**params[args.model])
    model.load_state_dict(torch.load(f"logdir/{args.model}/checkpoints/best.pth")['model_state_dict'])
    predictor = OneModelInference(model, path, vocab=vocab)

    predictor(F"{args.model}_submit.csv")