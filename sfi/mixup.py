import torch
import torch.nn as nn

import numpy as np


# Mixup for data augmentation
# https://arxiv.org/abs/1710.09412

class MixupDataLoaderAdaptor:
    def __init__(self, dataloader, alpha=0.4):
        self.dataloader = dataloader
        self.dataiter = None
        self.alpha = alpha

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.dataiter = iter(self.dataloader)
        return self

    def __next__(self):
        inputs1, labels1 = next(self.dataiter)

        n = inputs1.size(0)

        # draw t from (symmetric) beta distribution
        # take from one side to prevent duplicates

        t = np.random.beta(self.alpha, self.alpha, size=n)
        t = np.concatenate([t[:, None], 1 - t[:, None]], axis=1).max(axis=1)
        t = torch.FloatTensor(t)
        t = t.view(n, 1, 1, 1)

        # shuffle the batch inputs and targets to get second batch

        r = np.random.permutation(n)
        inputs2, labels2 = inputs1[r], labels1[r]

        # mix up the original batch with the shuffled batch

        inputs = t * inputs1 + (1 - t) * inputs2

        # With CrossEntropy we do not need the mixed up labels
        # labels = t * labels1.float() + (1 - t) * labels2.float()

        return inputs, t, labels1, labels2


class MixupCrossEntropyLossAdaptor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs, t, labels1, labels2):
        lhs = t * self.criterion(outputs, labels1)
        rhs = (1 - t) * self.criterion(outputs, labels2)
        return (lhs + rhs).mean()
