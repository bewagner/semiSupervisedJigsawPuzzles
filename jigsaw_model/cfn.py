import torch.nn as nn
import torch
from operator import mul
from functools import reduce

# from .alexnet import AlexNetImageNet

__all__ = ['CFN']


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class CFN(nn.Module):
    """
    This class defines the context free network.
    """

    def __init__(self, tile_size, n=3, number_of_classes=100, dropout=0.0, batch_norm=True, feature_extractor=None):
        """
        Define the network architecture.

        :param tile_size:
        :param batch_norm:
        :param dropout:
        :param feature_extractor: (Model) The network used to generate the features up to the fc6 layer.
        :param training: (Bool) Wether we are training.
        """
        super(CFN, self).__init__()

        self.num_elems_after_features = None
        self.batch_norm = batch_norm
        self.dropout = dropout

        assert isinstance(n, int)
        self.n = n

        assert isinstance(tile_size, tuple)
        self.tile_size = tile_size

        assert isinstance(number_of_classes, int)
        self.number_of_classes = number_of_classes

        if feature_extractor is None:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3),
                nn.BatchNorm2d(96) if self.batch_norm else nn.Dropout2d(p=0.0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3),

                nn.Conv2d(96, 256, kernel_size=2),
                nn.BatchNorm2d(256) if self.batch_norm else nn.Dropout2d(p=0.0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        else:
            self.features = feature_extractor

        # Calculate the input dimension for the fc6 layer
        self.set_num_elems_after_features()

        self.fc6 = nn.Sequential(
            nn.Linear(self.num_elems_after_features, 512),
            nn.BatchNorm1d(512) if batch_norm else nn.Dropout(p=0.0),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(n * n * 512, 4096),
            nn.BatchNorm1d(4096) if self.batch_norm else nn.Dropout(p=0.0),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.fc8 = nn.Sequential(
            nn.Linear(4096, self.number_of_classes)
        )

        self.apply(initialize_weights)

    def set_num_elems_after_features(self):
        """
        Calculate the number of elements that are output by the feature extractor. This is needed when setting up the
        fc6 layer.
        :return: None
        """
        feature_output = self.features(torch.rand((1, 3) + self.tile_size))
        self.num_elems_after_features = reduce(mul, feature_output.size()[1:])

    def evaluate_up_to_fc6(self, x):
        """
        Forward an input and get the output of fc6.

        :param x: (Tensor) Input [batch_dim channels height width]
        :return: (Tensor) Output of layer fc6
        """
        x = self.features(x)
        x = x.view(x.size(0), reduce(mul, x.size()[1:]))
        x = self.fc6(x)
        return x

    def forward(self, x):
        """
        Forward an input through the network

        :param x: (Tensor) Input [batch_dim tile_dim channels height width]
        :return: (Int) Classification output
        """
        # Get a list of individual tile batches
        tile_batches = torch.unbind(x, dim=1)

        # Run each tile batch through the first part of the net
        fc6_list = [self.evaluate_up_to_fc6(tile_batch) for tile_batch in tile_batches]

        # Concatenate all the results
        x = torch.cat(fc6_list, dim=1)

        # Apply the following layers
        x = self.fc7(x)
        x = self.fc8(x)
        return x
