import torch.nn as nn

__all__ = ['Flatten']


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
