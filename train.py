from models.bi_lstm import BiLSTM
import torch
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AUCCallback, F1ScoreCallback
import collections
from torch.utils.data import DataLoader
import os
from lib.dataset import TextDataset
import pandas as pd
from lib.vocab import CommentVocab

vocab = CommentVocab(load=True)
pad_size = 64

model = BiLSTM(vocab_size=len(vocab), pad_size=pad_size)
model = model

# experiment setup
logdir = "./logdir"
num_epochs = 20

# data
loaders = collections.OrderedDict()
data = pd.read_csv('data/train.csv')
test = data.sample(frac=0.2)
train = data.drop(test.index)
train, test = train.reset_index(), test.reset_index()
train_dataset = TextDataset(df=train, vocab=vocab, pad_size=pad_size)
val_dataset = TextDataset(df=test, vocab=vocab, pad_size=pad_size)
loaders["train"] = DataLoader(train_dataset, shuffle=True, batch_size=32)
loaders["valid"] = DataLoader(val_dataset, shuffle=False, batch_size=32)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

callbacks = None  # [AUCCallback(), F1ScoreCallback()]

# model runner
# Trainer(model, loaders).run()
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