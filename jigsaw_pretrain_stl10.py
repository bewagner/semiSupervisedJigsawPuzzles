from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple
import sys

# Custom imports
import logger
import preprocessing
import permutation
import jigsaw_model
import pipeline
import constants

torch.backends.cudnn.benchmark = True


def pretrain_stl10(number_of_epochs: int, experiment_description: str) -> Tuple[nn.Sequential, logger.Experiment]:
    """
    Train a jigsaw puzzle solver on the stl10 dataset

    :param experiment_description: Description of the target of the experiment
    :param number_of_epochs: Number of epochs to train
    :return:  model, logger
    """

    # region Set device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # endregion

    n = constants.n
    image_size = constants.image_size
    piece_crop_percentage = constants.piece_crop_percentage
    tile_size = constants.tile_size
    batch_size = 128
    lr = 10e-2
    batch_norm = True
    n_epochs = number_of_epochs
    use_visdom = True
    save_model = True

    # region Permutator
    permutator = permutation.Permutation(filename=Path("data", "permutations_d_1000.csv"))
    print("Permutator created.")
    # endregion

    # region Transform operations

    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomGrayscale(0.3),
        preprocessing.RandomJitterColorChannels(),
        transforms.RandomCrop(image_size, pad_if_needed=True),
        transforms.ToTensor(),
        preprocessing.CutIntoRandomTiles(image_size=image_size, piece_crop_percentage=piece_crop_percentage, max_rotation_angle=0),
        preprocessing.PerTileNormalize(),
        preprocessing.StackTiles()
    ])

    transforms_val = transforms.Compose([
        transforms.RandomCrop(image_size, pad_if_needed=True),
        transforms.ToTensor(),
        preprocessing.CutIntoDeterministicTiles(image_size=image_size, piece_crop_percentage=piece_crop_percentage),
        preprocessing.PerTileNormalize(),
        preprocessing.StackTiles()
    ])
    # endregion

    # region data sets

    image_path = Path("data", "stl10_images")
    images_train = jigsaw_model.STL10(root=image_path, transform=transforms_train, split='train+unlabeled')
    images_val = jigsaw_model.STL10(root=image_path, transform=transforms_val, split='test')

    dataset_train = jigsaw_model.JigsawTileDataset(dataset=images_train, n=n, permutator=permutator)
    dataset_val = jigsaw_model.JigsawTileDataset(dataset=images_val, n=n, permutator=permutator)
    # endregion

    # region Loaders
    loader_options = {'pin_memory': True, 'num_workers': 8, 'batch_size': batch_size}
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, **loader_options)
    loader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False, **loader_options)
    # endregion

    # region model
    number_of_classes = permutator.number_of_permutations
    model = jigsaw_model.CFN(number_of_classes=number_of_classes, batch_norm=batch_norm, tile_size=tile_size,
                             feature_extractor=jigsaw_model.AlexNetImageNet(batch_norm=batch_norm)).to(device)
    # endregion

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters())

    # region Logger
    log = pipeline.logger_from_params(use_visdom=use_visdom, dataset=image_path, learning_rate=lr,
                                      number_of_epochs=n_epochs, batch_size=batch_size, optimizer=optimizer,
                                      model=model, number_of_permutations=permutator.number_of_permutations,
                                      transforms=transforms_train, piece_crop_percentage=piece_crop_percentage,
                                      batch_norm=batch_norm, train_type='pretrain')

    pipeline.write_experiment_description(experiment_description, log)
    # endregion

    if not save_model:
        print("\n--- MODEL WILL NOT BE SAVED ---\n")

    for epoch in range(n_epochs):
        pipeline.train(loader_train, model, optimizer, device, epoch=epoch, criterion=criterion, log=log)
        current_accuracy = pipeline.validate(loader_val, model, device, epoch=epoch, criterion=criterion, log=log)

        if save_model:
            pipeline.maybe_save_checkpoint(current_accuracy, model, epoch, log=log)
        else:
            print("\nMODEL WILL NOT BE SAVED\n")

    return model, log


if __name__ == '__main__':
    pretrain_stl10(number_of_epochs=10, experiment_description="")
