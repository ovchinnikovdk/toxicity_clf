import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, lstm_output, last_state):
        hidden = last_state.permute(1, 2, 0)
        lstm_output = lstm_output.permute(0, 2, 1)
        attn_w = torch.bmm(lstm_output, hidden)
        soft_w = self.softmax(attn_w)
        return torch.bmm(lstm_output, soft_w.permute(0, 2, 1)).squeeze(2)


class BiLSTM_Attn(torch.nn.Module):
    def __init__(self, num_classes=6, pad_size=256, emb_size=512, vocab_size=3, n_layers=4, hidden_size=256):
        super(BiLSTM_Attn, self).__init__()
        self.emb_size = emb_size
        self.pad_size = pad_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.emb_size,
                                            scale_grad_by_freq=True)
        self.lstm = torch.nn.LSTM(input_size=self.emb_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=n_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.attn = Attention()
        self.fc = torch.nn.Sequential(torch.nn.Linear(2 * self.hidden_size * self.n_layers, 256),
                                      torch.nn.Dropout(0.4),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(256, num_classes))

    def forward(self, x):
        x, lengths = x
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, (last_hidden_state, last_cell_state) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=self.pad_size)
        attn_out = self.attn(x, last_hidden_state)
        x = attn_out.view(attn_out.shape[0], -1)
        x = self.fc(x)
        return x

