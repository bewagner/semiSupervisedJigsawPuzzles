from __future__ import print_function, division

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

import PIL
from PIL import Image, ImageDraw

import scipy
from scipy.spatial import Voronoi
from scipy.ndimage import measurements

from itertools import product
import math

import matplotlib.pyplot as plt
import matplotlib.patches

from typing import Tuple, List, Union
from scipy.ndimage import rotate as scipy_rotate

import constants

# Typing aliases
PILImage = PIL.Image.Image
integerList2D = List[List[int]]


class ResizeKeepingAspectRatio(object):
    """ Resize an image, keeping its aspect ratio, so that the smaller edge will have a given size."""

    def __init__(self, new_shorter_edge):
        """
        Constructor
        :param new_shorter_edge: Length of new shorter edge
        """
        assert isinstance(new_shorter_edge,
                          (int, float,
                           np.int32)), "Constructor of ResizeKeepingAspectRatio needs either float or integer input." \
                                       "Got %s." % type(new_shorter_edge)
        self.new_shorter_edge = float(new_shorter_edge)

    def __str__(self):
        return "ResizeKeepingAspectRatio({})".format(self.new_shorter_edge)

    def __call__(self, image):
        """
        This is where the resizing is done. First we calculate a resizing factor and the scale the input image.
        :param image: A PIL image
        :return: The resized PIL image
        """
        # Get the old height and width
        height, width = image.size

        # Calculate a resize factor to keep aspect ratio
        resize_ratio = max(self.new_shorter_edge / width, self.new_shorter_edge / height)

        # Calculate the new size of the image
        new_size = [int(math.ceil(resize_ratio * dimension)) for dimension in image.size]

        return image.resize(new_size, Image.LANCZOS)


class CutIntoRandomTiles(object):
    """ Cut an image into tiles and crop each tile at a random location."""

    def __init__(self, n: int = 3, piece_crop_percentage: float = 0.88, image_size: Tuple[int, int] = (99, 99),
                 max_rotation_angle: int = 0):
        """
        :param n: Number of tiles in each dimension. (That is the total number of tiles will be number_of_tiles^2)
        """
        assert isinstance(n,
                          int), "The input parameter 'number_of_tiles' for the constructor of CutIntoRandomTiles " \
                                "needs to be of type int." \
                                "Got %s." % type(n).__name__
        self.n = n

        assert isinstance(piece_crop_percentage,
                          float), "The input parameter 'piece_crop_percentage' for the constructor of " \
                                  "CutIntoRandomTiles needs to be of type float." \
                                  "Got %s." % type(piece_crop_percentage).__name__

        assert 1 >= piece_crop_percentage > 0, 'The input parameter \'piece_crop_percentage\' needs' \
                                               ' to be bigger than ' \
                                               'zero and smaller or equal to one.' \
                                               'Got %s.' % piece_crop_percentage

        self.piece_crop_percentage = piece_crop_percentage

        self.image_size = image_size[0]

        # Calculate the size of one image tile
        self.piece_size = int(self.image_size / self.n)

        # Define the locations of the sizes
        steps = range(0, self.image_size, self.piece_size)

        self.boxes = [(i, j, i + self.piece_size, j + self.piece_size) for i, j in product(steps, steps)]

        # Calculate size for randomly cropped out part
        self.cropped_piece_size = round(self.piece_crop_percentage * self.piece_size)

        self.max_rotate_angle = max_rotation_angle

    def __str__(self):
        return "CutIntoRandomTiles(n={},piece_crop_percentage={})".format(self.n, self.piece_crop_percentage)

    def __call__(self, image):
        """
        Cut the image into number_of_tiles^2 equally sized tiles.
        Then crop each of those tiles at random location for 85% of the size of the original tile.
        :param image: A PIL image
        :return: A list of number_of_tiles^2 tiles.
        """
        # Get the old height and width
        _, height, width = image.shape
        assert height == width, "CutIntoRandomTiles needs a quadratic image. " \
                                "Input size was (%d,%d)" % (width, height)
        assert height % self.n == 0, "CutIntoRandomTiles got an image that is not divisible into %d parts" % self.n

        # Crop out the random tensor tiles
        # if self.piece_crop_percentage < 1:
        return [self.random_crop(image[:, x_min:x_max, y_min:y_max]) for (x_min, y_min, x_max, y_max) in self.boxes]
        # else:
        #     return [image[:, x_min:x_max, y_min:y_max] for (x_min, y_min, x_max, y_max) in self.boxes]

    def random_crop(self, tensor):
        """
        Random crop a tensor to (self.cropped_piece_size, self.cropped_piece_size)
        :param tensor: Tensor
        :return: Cropped tensor
        """
        if self.max_rotate_angle > 0:
            tensor = rotate_image_tile(tensor, self.max_rotate_angle)

        x_shift = np.random.randint(0, tensor.shape[1] - self.cropped_piece_size)
        y_shift = np.random.randint(0, tensor.shape[1] - self.cropped_piece_size)

        return tensor[:, x_shift:self.cropped_piece_size + x_shift, y_shift:self.cropped_piece_size + y_shift]


class CutIntoDeterministicTiles(object):
    """ Cut an image into tiles and crop each tile at a random location."""

    def __init__(self, n=3, piece_crop_percentage=0.88, image_size=(99, 99)):
        """
        :param n: Number of tiles in each dimension. (That is the total number of tiles will be number_of_tiles^2)
        """
        assert isinstance(n,
                          int), "The input parameter 'number_of_tiles' for the constructor of " \
                                "CutIntoDeterministicTiles " \
                                "needs to be of type int." \
                                "Got %s." % type(n).__name__
        self.n = n

        assert isinstance(piece_crop_percentage,
                          float), "The input parameter 'piece_crop_percentage' for the constructor of " \
                                  "CutIntoDeterministicTiles needs to be of type float." \
                                  "Got %s." % type(piece_crop_percentage).__name__

        assert 1 >= piece_crop_percentage > 0, 'The input parameter \'piece_crop_percentage\' needs to be bigger ' \
                                               'than zero and smaller or equal to one.' \
                                               'Got %s.' % piece_crop_percentage

        self.piece_crop_percentage = piece_crop_percentage

        self.image_size = image_size[0]

        # Calculate the size of one image tile
        self.piece_size = int(self.image_size / self.n)

        # Define the locations of the sizes
        steps = range(0, self.image_size, self.piece_size)

        self.boxes = [(i, j, i + self.piece_size, j + self.piece_size) for i, j in product(steps, steps)]

        # Calculate size for randomly cropped out part
        self.cropped_piece_size = round(self.piece_crop_percentage * self.piece_size)

    def __str__(self):
        return "CutIntoDeterministicTiles(n={},piece_crop_percentage={})".format(self.n, self.piece_crop_percentage)

    def __call__(self, image):
        # TODO Comment

        _, height, width = image.shape
        assert height == width, "CutIntoDeterministicTiles needs a quadratic image. " \
                                "Input size was (%d,%d)" % (width, height)
        assert height % self.n == 0, "CutIntoDeterministicTiles got an image that is" \
                                     " not divisible into %d parts" % self.n

        # Crop out the tensor tiles
        if self.piece_crop_percentage < 1:
            return [self.random_crop(image[:, x_min:x_max, y_min:y_max]) for (x_min, y_min, x_max, y_max) in self.boxes]
        else:
            return [image[:, x_min:x_max, y_min:y_max] for (x_min, y_min, x_max, y_max) in self.boxes]

    def random_crop(self, tensor):
        """
        Crop a tensor to (self.cropped_piece_size, self.cropped_piece_size)
        :param tensor: Tensor
        :return: Cropped tensor
        """
        x_shift = (self.piece_size - self.cropped_piece_size) // 2
        y_shift = (self.piece_size - self.cropped_piece_size) // 2
        return tensor[:, x_shift:self.cropped_piece_size + x_shift, y_shift:self.cropped_piece_size + y_shift]


def channel_mean(image):
    """
    Get the per channel mean of an image
    :param image: Tensor Height x Width x Channels
    :return: Tensor of shape [Channels]
    """
    return image.contiguous().view(image.size(0), -1).mean(-1)


def channel_std(image):
    """
        Get the per channel standard deviation of an image
        :param image: Tensor Height x Width x Channels
        :return: Tensor of shape [Channels]
        """
    return image.contiguous().view(image.size(0), -1).std(-1)


def rotate_image_tile(tile: torch.Tensor, max_angle: int) -> torch.Tensor:
    angle = np.random.randint(-max_angle, max_angle + 1)
    rotated = scipy_rotate(tile, angle=angle, axes=(2, 1), mode='reflect')
    return torch.from_numpy(rotated)


def rotate_image_tiles(tiles: List[torch.Tensor], max_angle: int) -> List[torch.Tensor]:
    for i, tile in enumerate(tiles):
        tiles[i] = rotate_image_tile(tile, max_angle)

    return tiles


class PerTileToTensor(object):
    """ Make a list of images into tensors """

    def __str__(self):
        return "PerTileToTensor()"

    def __call__(self, tiles):
        """
        :param tiles: A list of PIL images
        :return: A list of tensors
        """
        return [transforms.ToTensor()(image) for image in tiles]


class CutImageIntoVoronoiPolygonTensors(object):
    """ Cut an image into voronoi polygons """

    # TODO Problem: Puzzle wird nicht so sortiert, wie ein Mensch es machen würde.
    #  Das könnte eventuell durch Clustern der Punkte erreicht werden.
    def __init__(self, max_rotation_angle: int = 0):
        self.to_tensor_transformer = transforms.ToTensor()
        self.max_rotation_angle = max_rotation_angle

    def __str__(self):
        return "CutImageIntoVoronoiPolygonTensors()"

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            raise ValueError(
                "Input passed to CutImageIntoVoronoiPolygonTensors() was of type torch.Tensor. "
                "Remember that this transform expects PIL image input.")

        polygon_images = cutout_image_polygons(image, self.max_rotation_angle)

        return [self.to_tensor_transformer(image) for image in polygon_images]


def voronoi_finite_polygons_2d(vor: scipy.spatial.Voronoi, radius: float = None) -> Tuple[
    integerList2D, np.ndarray]:
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def cutout_image_polygons(image: Union[PILImage, np.ndarray], max_angle: int, plot_image: bool = False) -> List[
    PILImage]:
    """
    Cut an image into voronoi polygons

    First we distribute random points on the picture. Then we create the voronoi polygons of these points and
    cut the image into parts according to these polygons. Finally each image is resized to constants.image_size.
    :param max_angle: Maximum rotation angle
    :param plot_image:  Wether to plot the voronoi to an image
    :param image: Input image
    :return: A list of n x n images
    """

    image_size = constants.image_size
    n = constants.n

    points = np.random.randint(low=0, high=image_size[0], size=(n * n, 2))
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    polygon_images = []
    points = np.empty((n * n, 2))
    for i, region in enumerate(regions):
        polygon = [tuple(x) for x in vertices[region]]

        # Create a masked image
        image_array = np.asarray(image)
        mask_image = Image.new('L', (image_array.shape[1], image_array.shape[0]), 0)
        ImageDraw.Draw(mask_image).polygon(polygon, outline=1, fill=1)

        mask = np.array(mask_image)
        image_array = image_array * np.stack([mask, mask, mask], axis=2)

        # Compute the center of mass for the mask. This is the center point for the region and will help us calculate
        # the index of the region later
        center = measurements.center_of_mass(mask)
        points[i, 0] = center[1]
        points[i, 1] = center[0]

        # Crop the image to the masked part
        coords = np.argwhere(image_array.sum(axis=2) > 0)

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        cropped = image_array[x_min:x_max + 1, y_min:y_max + 1]

        # Append the resulting image
        polygon_image = Image.fromarray(cropped, "RGB").resize(constants.tile_size)
        if max_angle > 0:
            angle = np.random.randint(-max_angle, max_angle + 1)
            polygon_image = polygon_image.rotate(angle, expand=False)

        polygon_images.append(polygon_image)

    ind = np.lexsort((points[:, 0], -points[:, 1]))

    # Sort the image tiles from top to bottom and left to right
    polygon_images = [polygon_images[i] for i in ind]

    if plot_image:
        points = points[ind]
        plot_color = '#c2d2e9'
        # colorize
        ax = plt.gca()
        for region in regions:
            polygon = vertices[region]
            ax.add_patch(matplotlib.patches.Polygon(polygon,closed=True,fill=False, color=plot_color))
            # plt.fill(*zip(*polygon), alpha=0.6)


        for i, point in enumerate(points):
            plt.plot(point[0], point[1], 'o', color=plot_color)
            # ax.annotate(i, (point[0], point[1]))
        # plt.xlim(0, 225)
        # plt.ylim(0, 225)

        plt.axis('off')


        plt.imshow(image, aspect='auto')
        plt.tight_layout()
        plt.savefig('plot.png')
        plt.show()

    return polygon_images


class PerTileNormalize(object):
    """ Normalize a list of tile tensors """

    def __str__(self):
        return "PerTileNormalize()"

    def __call__(self, tiles):
        """
        :param tiles: A list of tensors
        :return: A list of normalized tile tensors
        """
        output = []
        for tile in tiles:
            mean = channel_mean(tile)
            std = channel_mean(tile)

            # If the standard deviation is zero, we will set it to 1 to prevent nan values caused by division by zero.
            std[std == 0] = 1

            # Normalize the tensor
            tile_tensor_normalized = transforms.Normalize(mean, std)(tile)

            output.append(tile_tensor_normalized)
        return output


class StackTiles(object):
    """ Stack a list of tensors into one Tensor """

    def __str__(self):
        return "StackTiles()"

    def __call__(self, tiles):
        """
        :param tiles: A list of tensors
        :return: A tensor of the stacked tiles. Dimension: [tile channels height width]
        """
        return torch.stack(tiles)


class RandomJitterColorChannels(object):
    """ Randomly jitter an RGB-images color channels. """

    def __init__(self, max_shift=2):
        """
        :param max_shift: Maximum number of pixels that we want to shift the color channels in each direction.
        """
        assert isinstance(max_shift, int), "RandomJitterColorChannels needs an int as input for max_shift." \
                                           " Got a %s." % type(max_shift)
        assert max_shift > -1, "RandomJitterColorChannels parameter max_shift needs to be bigger than -1." \
                               " Was %d." % max_shift
        self.max_shift = max_shift

    def __str__(self):
        return "RandomJitterColorChannels({})".format(self.max_shift)

    def __call__(self, image):
        """
        First we extract the individual channels from each image. Then each of those channels is
        shifted by a random amount.
        Empty pixels are filled up with zeros. Finally the shifted channels will be recombined to a picture.
        :param image: A PIL image
        :return: A PIL image, in which all the color channels were randomly shifted
        """
        # Extract the image channels and the image format
        channels = image.split()
        height, width = channels[0].size

        # If we have grayscale input, there is only one color channel, so shifting the channels makes no sense.
        if len(channels) == 1:
            return image

        output = np.zeros((width, height, len(channels)), 'uint8')

        for i, channel in enumerate(channels):
            channel_array_padded = np.pad(np.array(channel),
                                          [[self.max_shift, self.max_shift], [self.max_shift, self.max_shift]],
                                          mode='reflect')

            shift_x = np.random.randint(0, 2 * self.max_shift)
            shift_y = np.random.randint(0, 2 * self.max_shift)

            channel_shifted = channel_array_padded[shift_x: shift_x + width, shift_y:shift_y + height]
            output[:, :, i] = channel_shifted

        return Image.fromarray(output)
