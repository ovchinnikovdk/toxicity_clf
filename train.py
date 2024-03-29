from models.bi_lstm import BiLSTM
from models.bi_lstm_attn import BiLSTM_Attn
from models.cnn import TextCNN
import torch
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AUCCallback, F1ScoreCallback
import collections
from torch.utils.data import DataLoader
from lib.dataset import TextTrainDataset, TrainPreparedDataset
import pandas as pd
from lib.vocab import CommentVocab
import argparse

# vocab = CommentVocab(load=True)
pad_size = 256

models = {
    'bilstm': BiLSTM,
    'bilstm_attn': BiLSTM_Attn,
    'text_cnn': TextCNN

}
params = {
    'bilstm': {
        'vocab_size': 10003,
        'pad_size': pad_size
    },
    'bilstm_attn': {
        'vocab_size': 10003,
        'pad_size': pad_size
    },
    'text_cnn': {
        'vocab_size': 10003,
        'pad_size': pad_size
    }
}


def train(num_epochs, model, loaders, logdir):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    callbacks = [AUCCallback(), F1ScoreCallback()]

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        callbacks=callbacks,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help='Select model to train, for all (stacking) use \'all\'')
    parser.add_argument("--n_epochs", type=int, default=30, help='Number of epochs to train')

    args = parser.parse_args()

    if args.model == 'all':
        raise NotImplementedError
    model = models[args.model](**params[args.model])

    # experiment setup
    logdir = "./logdir/" + args.model
    num_epochs = args.n_epochs

    # data
    loaders = collections.OrderedDict()
    data = pd.read_csv('data/train_prepared.csv')
    test_df = data.sample(frac=0.1)
    train_df = data.drop(test_df.index)
    train_df, test = train_df.reset_index(), test_df.reset_index()
    train_dataset = TrainPreparedDataset(train_df)
    val_dataset = TrainPreparedDataset(test_df)
    loaders["train"] = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=160)
    loaders["valid"] = DataLoader(val_dataset, shuffle=False, num_workers=4, batch_size=160)

    train(num_epochs, model, loaders, logdir)


if __name__ == '__main__':
    main()
