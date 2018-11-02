from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple
import logger

# Custom imports
import helpers
import jigsaw_model
import pipeline
import constants
import preprocessing

torch.backends.cudnn.benchmark = True

def train_classify_stl10(number_of_epochs: int, train_type: str, model_path: Path, load_model: bool = False,
                         freeze_weights: bool = True) \
        -> Tuple[nn.Sequential, logger.Experiment]:
    """
    Train a model to classify STL10 images.

    If load_model==True, a pretrained model is loaded. Then a new fully connected head is appended to this pretrained
    model and this model is trained with the CNN-layers frozen.

    Else, The model is loaded with random weights.

    :param freeze_weights: Wether to freeze the first convolutional layers
    :param number_of_epochs: Number of epochs to train.
    :param train_type: Name of the training type. Is used as a name for logging.
    :param model_path: The path to where the model is saved (and/or loaded).
    :param load_model: Wether to load a pretrained model.
    :return: The model and the training logger.
    """
    # region Set device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # endregion

    image_size = constants.image_size
    batch_size = 128
    lr = 10e-2
    n_epochs = number_of_epochs
    use_visdom = True
    save_model = True

    # region Transform operations
    normalize = transforms.Normalize(mean=constants.mean,
                                     std=constants.stdDev)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(image_size, pad_if_needed=True),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.RandomCrop(image_size, pad_if_needed=True),
        transforms.ToTensor(),
        normalize
    ])
    # endregion

    # region data sets
    image_path = Path("data", "stl10_images")
    images_train = jigsaw_model.STL10(root=image_path, transform=transform_train, split='train')
    images_val = jigsaw_model.STL10(root=image_path, transform=transform_val, split='test')
    # endregion

    # region Loaders
    loader_options = {'pin_memory': True, 'num_workers': 8, 'batch_size': batch_size}
    loader_train = torch.utils.data.DataLoader(images_train, shuffle=True, **loader_options)
    loader_val = torch.utils.data.DataLoader(images_val, shuffle=False, **loader_options)
    # endregion

    # region Model
    cnn_model = jigsaw_model.CFN(number_of_classes=100, batch_norm=True, tile_size=constants.tile_size,
                                 feature_extractor=jigsaw_model.AlexNetImageNet(batch_norm=True)).to(device)
    if load_model:
        print("Loaded model")
        load_path = Path("Experiments").joinpath(model_path)
        cnn_model = pipeline.load_weights(cnn_model, load_path, 'pretrain')

    # Freeze all the weights for the convolutional filters
    if freeze_weights:
        cnn_model = helpers.freeze_first_n_layers(cnn_model, 5)

    top = [jigsaw_model.Flatten(),
           nn.Linear(256 * 6 * 6, 4096),
           nn.ReLU(inplace=True),
           nn.Linear(4096, 4096),
           nn.ReLU(inplace=True),
           nn.Linear(4096, constants.stl10_number_of_classes)]

    cnn_model = helpers.add_top_to_cfn(cnn_model, top).to(device)
    # endregion Model

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, cnn_model.parameters()), lr=lr, weight_decay=0.0)

    # region Logger
    log = pipeline.logger_from_params(use_visdom=use_visdom, dataset=image_path, learning_rate=lr,
                                      number_of_epochs=n_epochs, batch_size=batch_size, optimizer=optimizer,
                                      model=cnn_model, transforms=transform_train, train_type=train_type)
    # Overwrite logger path
    log.name = str(model_path.parts[-1])
    # endregion

    if not save_model:
        print("\n--- MODEL WILL NOT BE SAVED ---\n")

    for epoch in range(n_epochs):
        pipeline.train(loader_train, cnn_model, optimizer, device, epoch=epoch, criterion=criterion, log=log)
        current_accuracy = pipeline.validate(loader_val, cnn_model, device, epoch=epoch, criterion=criterion, log=log)

        if save_model:
            pipeline.maybe_save_checkpoint(current_accuracy, cnn_model, epoch, log=log)
        else:
            print("\nMODEL WILL NOT BE SAVED\n")

    return cnn_model, log



