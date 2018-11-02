from unittest import TestCase
from pathlib import Path
from PIL import Image, ImageChops
import preprocessing
import numpy as np
from torchvision import transforms
import torch


class TestPreprocessing(TestCase):
    """
    Class to test image preprocessing.
    """

    def setUp(self):
        """
        Load a test image
        """
        image_path = Path('dog.jpg')
        self.image = Image.open(image_path)

    def test_RandomToGreyscale(self):
        """
        Test the random to greyscale transformation
        :return: None
        """
        t = preprocessing.RandomToGreyscale(1.0)
        image_transformed = t(self.image)

        # Check if image has the right mode
        self.assertEqual('RGB', image_transformed.mode)

        # Check if image has still the old size
        self.assertEqual(self.image.size, image_transformed.size)

        # Check if the image content is the same
        self.assertTrue(
            ImageChops.difference(image_transformed.convert('L'), self.image.convert('L')).getbbox() is None)

    def test_ResizeKeepingAspectRatio(self):
        for new_shorter_edge in np.random.randint(4, 3000, size=6):
            t = preprocessing.ResizeKeepingAspectRatio(new_shorter_edge=new_shorter_edge)
            image_transformed = t(self.image)

            # Test if the image has the required minimum size
            self.assertTrue(any(i >= new_shorter_edge for i in image_transformed.size))

            # Test if the aspect ratio stayed correct
            height, width = self.image.size
            ratio = height / width
            height_transformed, width_transformed = image_transformed.size
            ratio_transformed = height_transformed / width_transformed
            self.assertAlmostEqual(ratio, ratio_transformed, delta=0.2)

            # Test if images smaller than the new_shorter_edge get rescaled correctly
            image_small = self.image.resize((int(new_shorter_edge / 2), int(new_shorter_edge / 2)))
            self.assertTrue(any(i >= new_shorter_edge for i in t(image_small).size))

            # Test if image stays correct when it already has the right size
            image_correct = self.image.resize((new_shorter_edge, new_shorter_edge))
            self.assertTrue(ImageChops.difference(image_correct, t(image_correct)).getbbox() is None)

            # Check if image has the right mode
            self.assertEqual('RGB', image_transformed.mode)

    def test_RandomJitterColorChannels(self):
        """
        Test the random color channel jittering
        :return: None
        """
        t = preprocessing.RandomJitterColorChannels(max_shift=2)
        image_transformed = t(self.image)

        # Check if image has the right mode
        self.assertEqual('RGB', image_transformed.mode)

        # Check if image has still the old size
        self.assertEqual(self.image.size, image_transformed.size)

    def test_CutImageIntoRandomTiles(self):
        square_image_size = 300
        center_cropper = transforms.CenterCrop(square_image_size)
        square_image = center_cropper(self.image)

        n = 3
        piece_crop_percentage = 0.8
        t = preprocessing.CutImageIntoRandomTiles(n=n, piece_crop_percentage=piece_crop_percentage)
        tiles = t(square_image)

        # Test if correct number of tiles was cut out
        self.assertEqual(n * n, len(tiles))

        # Test if tiles have correct size
        self.assertTrue(all(tile.size == (80, 80) for tile in tiles))

        # Test if tile mode is correct
        self.assertTrue(all(tile.mode == 'RGB' for tile in tiles))

    def test_PerTileNormalize(self):

        # Generate random data with mean 0.5 and std 20
        tensor_transformer = transforms.ToTensor()
        a = tensor_transformer(Image.fromarray(np.random.normal(0.5, 20, (120, 120))))
        b = tensor_transformer(Image.fromarray(np.random.normal(0.5, 20, (120, 120))))
        c = tensor_transformer(Image.fromarray(np.random.normal(0.5, 20, (120, 120))))
        tiles = [a, b, c]

        # Check if data has mean 0 and std 1 after normalizing
        t = preprocessing.PerTileNormalize()
        for tile in t(tiles):
            self.assertAlmostEqual(torch.mean(tile), 0, delta=0.01)
            self.assertAlmostEqual(torch.std(tile), 1, delta=0.01)


