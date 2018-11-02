from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import permutation
import pathlib

__all__ = ['STL10']


class STL10(Dataset):
    """
    Custom wrapper for the rescaled STL10 dataset
    """

    def __init__(self, root, transform=None, split='train+unlabeled'):

        if not isinstance(split, str) or split not in ['train', 'test', 'unlabeled', 'train+unlabeled']:
            raise ValueError(
                "Invalid argument for split: " + split + "\nValid choices are: unlabeled, test, train, train+unlabeled")

        self.split = split
        self.root = root
        self.train_plus_unlabeled = self.split == 'train+unlabeled'
        self.transform = transform

        if self.train_plus_unlabeled:
            self.dataset = [datasets.ImageFolder(root=str(root.joinpath('train').resolve()), transform=transform),
                            datasets.ImageFolder(root=str(root.joinpath('unlabeled').resolve()), transform=transform)]
            self.classes = self.dataset[0].classes
        else:
            self.dataset = datasets.ImageFolder(root=str(root.joinpath(split).resolve()), transform=transform)
            self.classes = self.dataset.classes

    def __len__(self):
        """
        Get the length of the dataset
        :return: Length (int)
        """
        if self.train_plus_unlabeled:
            return len(self.dataset[0]) + len(self.dataset[1])
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        """
        Get an image and a lable from the dataset.

        Attention: If split == 'train+unlabeled' the returned label will not be correct for the images from the
        unlabeled set
        :param index: Index of the image to return
        :return: (Image, Index of the images label)
        """
        if self.train_plus_unlabeled:
            if index < len(self.dataset[0]):
                return self.dataset[0][index]
            else:
                return self.dataset[1][index - len(self.dataset[0])]
        else:
            return self.dataset[index]
