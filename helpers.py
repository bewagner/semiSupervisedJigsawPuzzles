from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import PIL
from PIL import Image
import math
import collections
import numpy as np
import sys
from typing import Union, List, Tuple

import preprocessing


def visualize(image_input: Union[np.ndarray, torch.Tensor, Image.Image]):
    """
    Helper method to visualize all kinds of image data
    :param image_input: Input to visualize
    :return: None
    """

    if not isinstance(image_input, torch.Tensor) and not isinstance(image_input, Image.Image) and not isinstance(
            image_input, collections.Iterable):
        raise ValueError(
            "Function 'visualize' needs a tensor or a PIL image as image_input. Got: %s." % type(image_input).__name__)

    # Handling iterable image_input
    if isinstance(image_input, collections.Iterable) and not isinstance(image_input, torch.Tensor):
        if all(isinstance(image, Image.Image) for image in image_input):
            image_input = [transforms.ToTensor()(image) for image in image_input]
            visualize(torch.stack(image_input))
            return

        if all(isinstance(image, torch.Tensor) for image in image_input):
            visualize(torch.stack(image_input))
            return

        else:
            raise ValueError("Visualize got iterable image_input of mixed types.")

    # If we got a PIL image, we directly display it
    if isinstance(image_input, Image.Image):
        plt.imshow(image_input)
        plt.show()
        return

    # If we got a tensor, we check the number of dimensions:
    if isinstance(image_input, torch.Tensor):
        number_of_dimensions = len(image_input.size())

        if number_of_dimensions < 3:
            raise ValueError("Dimension of image_input tensor was too small to visualize.")

        # Three dimensions are assumed to be [channel height width], so we transform the tensor to an image and display
        if number_of_dimensions == 3:
            plt.imshow(transforms.ToPILImage()(image_input))
            plt.show()
            return

        # Four dimensions are assumed to be [tile channel height width], so we display all the tiles
        if number_of_dimensions == 4:

            if math.sqrt(image_input.size()[0]) % 1 != 0:
                raise ValueError("In visualize: The number of tiles was not a square.")

            tiles = [transforms.ToPILImage()(tile) for tile in torch.unbind(image_input)]

            n = int(math.sqrt(image_input.size()[0]))

            f, axes_array = plt.subplots(n, n)

            for j in range(n):
                for i in range(n):
                    axes_array[j, i].imshow(tiles[n * j + i], aspect='auto')
                    axes_array[j, i].axis('off')
                    axes_array[j, i].autoscale = False
            plt.subplots_adjust(wspace=0.001, hspace=0.001)
            plt.show()

        else:
            visualize(image_input[0, ...])


def get_batch(data_loader: DataLoader) -> torch.Tensor:
    """ Get a single batch from a data loader."""
    return next(iter(data_loader))


def cross_entropy_loss_for_random_prediction(number_of_classes: int) -> float:
    """ Calculate the cross entropy loss for a random distributed prediction """
    if number_of_classes <= 0:
        raise ValueError("Number of classes needs to be bigger than zero.")

    return - np.log(1 / number_of_classes)


def show_nth_from_loader(loader: DataLoader, n: int = 0, invert_permutation: bool = True):
    batch, permutations = get_batch(loader)

    image_data = batch[n, ...]
    permutation_index = permutations[n]

    if invert_permutation:
        inverse_permutation = loader.dataset.permutator.inverse_permutation(
            loader.dataset.permutator.permutations[permutation_index])
        image_data = image_data[inverse_permutation, :, :, :]

    visualize(image_data)


def show_nth_from_dataset(dataset: Dataset, n: int = 0, invert_permutation: bool = True):
    images, permutation_index = dataset[n]
    if invert_permutation:
        inverse_permutation = dataset.permutator.inverse_permutation(dataset.permutator.permutations[permutation_index])
        images = images[inverse_permutation, :, :, :]
    visualize(images)


def print_tensor_summary(t: torch.Tensor):
    if not isinstance(t, torch.Tensor):
        print("Input to print_tensor_summary was no tensor.")
        return
    t = t.type(torch.DoubleTensor)

    values = {'shape': tuple(t.shape), 'min': torch.min(t), 'max': torch.max(t), 'mean': torch.mean(t),
              'std': torch.std(t)}

    print("Tensor of shape {shape}\n"
          "Min \t\t{min:.3f}\n"
          "Max \t\t{max:.3f}\n"
          "Mean \t\t{mean:.3f}\n"
          "Std \t\t{std:.3f}".format(**values))
    return


def to_camel_case(snake_str: str) -> str:
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + ''.join(x.title() for x in components[1:])


def contains_nan(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor contains nan values
    :param tensor: Tensor
    :return: Wether the tensor contains nan values
    """
    return (tensor != tensor).nonzero().size(0) > 0


def query_yes_no(question: str, default: str = "yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def rescale_images(path, size):
    """
    Rescale all images in a directory to a square of given size
    :param path: Path where the images reside
    :param size: Size the images will be resized to
    :return: None
    """
    transformer = transforms.Compose([preprocessing.ResizeKeepingAspectRatio(size), transforms.CenterCrop(size)])

    for file in path.glob('*'):
        image = Image.open(file)
        image = transformer(image)
        image.save(file)


def freeze_first_n_layers(cnn_model: nn.Sequential, n: int) -> nn.Sequential:
    """
    Freeze the first n layers of the model
    :param cnn_model: The neural network model
    :param n: Number of layers to freeze
    """
    if n < 0:
        raise ValueError("The parameter n needs to be bigger than zero.")

    features = cnn_model.features.features
    counter = 0

    for layer in features:
        if str(layer).startswith('Conv'):
            counter += 1

        if not counter - 1 < n:
            break

        for param in layer.parameters():
            param.requires_grad = False
    return cnn_model


def print_net_frozen_features_summary(cnn_model: nn.Sequential):
    """
    Print a summary of the neural nets features.

    This can be used to check wether layers really get frozen correctly
    :param cnn_model: Model
    """
    features = cnn_model.features.features
    frozen = " FROZEN "
    active = " Active "

    for i, layer in enumerate(features):
        if str(layer).startswith("Conv") and i > 0:
            print("-" * 39)

        output = "{:15}\t".format(str(layer).split("(")[0])

        if hasattr(layer, 'weight'):
            output += " W: "
            if layer.weight.requires_grad:
                output += active
            else:
                output += frozen

        if hasattr(layer, 'bias'):
            output += " B: "
            if layer.weight.requires_grad:
                output += active
            else:
                output += frozen

        print(output)


def add_top_to_cfn(cfn, top):
    """
    Add a top of fully connected layers to the feature extractors of a CFN model
    :param cfn: Context free network
    :param top: List of nn.Modules to add to the network
    :return: Model
    """
    model = cfn.features.features
    module_list = list(model)
    return nn.Sequential(*(module_list + top))
