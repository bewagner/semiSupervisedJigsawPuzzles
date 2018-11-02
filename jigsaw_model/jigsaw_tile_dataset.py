from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import permutation
import pathlib

__all__ = ['JigsawTileDataset']


class JigsawTileDataset(Dataset):
    """ Wrapper class that takes the pictures from a dataset, cuts them into tiles and permutates those tiles."""

    def __init__(self, dataset=None, root_dir=None, transform=None, permutator=None, n=3):
        """
        Set all the necessary members.


        :param root_dir: The root directory in which the images reside. (Required if dataset == None)
        :param permutator: A Permutator object
        :param transform: Transforms to applay to the images.
        """
        if dataset is not None and not isinstance(dataset, str):
            self.root_dir = dataset.root
            self.data = dataset
        elif (isinstance(root_dir, str) or issubclass(type(root_dir), pathlib.Path)) and transform is not None:
            self.root_dir = root_dir
            self.transform = transform
            self.data = datasets.ImageFolder(self.root_dir, transform=self.transform)
        else:
            raise ValueError("JigsawTileDataset need valid dataset or root_dir and transform input.")

        if isinstance(permutator, permutation.Permutation):
            self.permutator = permutator
        else:
            self.permutator = permutation.Permutation(number_of_tiles=n * n)

        self.n = n

    def __len__(self):
        """
        Get the count of images in the dataset.
        :return: (Int) Number of images in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an image from the underlying dataset and a permutation. Then cut the image into tiles and apply the
        permutation.
        :param idx: (Int) Index of the image
        :return: (Dict) The image tiles and the corresponding permutation.
        """

        tiles = self.data[idx][0]

        # Get a random current_permutation
        current_permutation = self.permutator.randomly_choose_permutation()

        # Permutate by the given current_permutation
        tiles = tiles[current_permutation['permutation'], :, :, :]

        return tiles, current_permutation['index']
