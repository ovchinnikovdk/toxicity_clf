import torch


class StackingModels:
    def __init__(self, models=[]):
        self.models = models
        self.fc = torch.nn.Sequential(torch.nn.Linear(6 * len(models), 128),
                                      torch.nn.Dropout(0.4),
                                      torch.nn.Linear(128, 6),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        x = torch.cat(outputs, dim=0)
        x = self.fc(x)
        return x
