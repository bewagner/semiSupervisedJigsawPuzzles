import torch.nn as nn

__all__ = ['AlexNetImageNet', 'AlexNetSmall']


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

    def forward(self, x):
        """
        Forward an input through the model.
        :param x: (Tensor [batch_size channels height width]) Input that will be forwarded through the model
        :return: (Tensor) Output of the model
        """

        return self.features(x)


class AlexNetImageNet(AlexNet):
    def __init__(self, batch_norm):
        """
        The classic AlexNet used in the ILSVRC2012 challenge
        """
        super(AlexNetImageNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64) if batch_norm else nn.Dropout2d(p=0.0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192) if batch_norm else nn.Dropout2d(p=0.0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384) if batch_norm else nn.Dropout2d(p=0.0),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batch_norm else nn.Dropout2d(p=0.0),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batch_norm else nn.Dropout2d(p=0.0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


class AlexNetSmall(AlexNetImageNet):
    def __init__(self, batch_norm, n=2):
        """
        The same as the ImageNet version of AlexNet but without the last Max-Pooling layer.
        """
        super(AlexNetSmall, self).__init__(batch_norm)
        while n > 0:
            name = self.features[-1]._get_name()
            if name.__contains__("MaxPool") or name.__contains__("Conv2d"):
                n -= 1
            self.features.__delitem__(len(self.features) - 1)
