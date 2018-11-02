import math
from unittest import TestCase

from permutation import Permutation


class TestPermutation(TestCase):
    """
    Class to test the permutation class.
    """

    def setUp(self):
        """
        Set up the permutation objects for testing.
        """
        self.p = Permutation()

    def test___init__(self):
        """
        Test the constructor of the Permutation class
        """
        # Test wether the expected exceptions are thrown
        self.assertRaises(ValueError, lambda: Permutation(number_of_tiles=0))  # number_of_tiles too small
        self.assertRaises(ValueError, lambda: Permutation(number_of_tiles=10))  # number_of_tiles too big

        # number_of_permutations too small
        self.assertRaises(ValueError, lambda: Permutation(number_of_permutations=0))
        # number_of_permutations too big
        self.assertRaises(ValueError, lambda: Permutation(number_of_permutations=math.factorial(10)))

        # Criterion not callable
        self.assertRaises(ValueError, lambda: Permutation(criterion=2))

        # --- Test if the permutations are generated correctly ---
        # Test that the indices are correct
        indices = set(perm['index'] for perm in self.p.permutations)
        self.assertEquals(set(range(self.p.number_of_permutations)), indices)

        # Test that each permutation contains the correct values
        self.assertTrue(all(set(perm['permutation']) == set(range(9)) for perm in self.p.permutations))


    def test_calculate_statistics(self):
        """
        Test the method that calculates statistics about the selected permutation set.
        """
        # Calculate statistics
        min_d, max_d, mean_d, intra_list_dist_mean = self.p.calculate_statistics()
        self.assertEqual(max_d, 9)
        self.assertGreater(mean_d, 8.5)
        self.assertGreater(intra_list_dist_mean, 7.5)

    def test_randomly_choose_permutation(self):
        """
        Test the random permutation selection
        :return:
        """

        # Test if the permutation contains the correct values
        for i in range(100):
            random_permutation = self.p.randomly_choose_permutation()
            self.assertEqual(set(random_permutation['permutation']), set(range(9)))

    def test_hamming_distance(self):
        """
        The the function calculating the hamming distance
        :return:
        """
        self.assertEqual(self.p.hamming_distance([], []), 0)

        for length in range(160):

            l = list(range(length))
            l_copy = l.copy()
            counter = 0

            for i in range(int(math.floor(length / 2 - 1))):
                l[i], l[length - i - 1] = l[length - i - 1], l[i]
                counter = counter + 2
                self.assertEqual(self.p.hamming_distance(l, l_copy), counter)
