import torch


class TextCNN(torch.nn.Module):
    def __init__(self, num_classes=6, pad_size=256, emb_size=512, vocab_size=3):
        super(TextCNN, self).__init__()
        self.emb_size = emb_size
        self.pad_size = pad_size
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.emb_size,
                                            scale_grad_by_freq=True)
        self.cnn = torch.nn.Sequential(torch.nn.Conv2d(1, 8, kernel_size=5),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool2d(kernel_size=2),
                                       torch.nn.BatchNorm(8),
                                       torch.nn.Conv2d(8, 8, kernel_size=5),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool2d(kernel_size=2),
                                       torch.nn.BatchNorm(8))
        self.fc = torch.nn.Sequential(torch.nn.Linear(2 * (self.emb_size - 5) * (self.pad_size - 5), 512),
                                      torch.nn.Dropout(0.5),
                                      torch.nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.embedding(x)
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
