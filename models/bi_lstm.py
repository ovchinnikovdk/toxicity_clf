import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(torch.nn.Module):
    def __init__(self, num_classes=6, pad_size=256, emb_size=512, vocab_size=3, n_layers=4, hidden_size=128):
        super(BiLSTM, self).__init__()
        self.emb_size = emb_size
        self.pad_size = pad_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = 4
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.emb_size,
                                            scale_grad_by_freq=True)
        self.lstm = torch.nn.LSTM(input_size=self.emb_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=n_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.pool = torch.nn.AvgPool2d(kernel_size=5)
        linear_size = ((pad_size - 5) // 5 + 1) * ((2 * self.hidden_size - 5) // 5 + 1)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(linear_size, 256),
                                              torch.nn.Dropout(0.4),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(256, num_classes))

    def forward(self, x):
        x, lengths = x
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, h = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=self.pad_size)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
