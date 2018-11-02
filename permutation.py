import csv
import itertools
import math
import os
import pathlib
import random

import numpy as np

"""
Class that manages the permutations of the image tiles.
"""


class Permutation:

    def __init__(self, number_of_tiles: int = 9, number_of_permutations: int = 100, criterion: object = max,
                 seed: object = None, filename: pathlib.Path = None, dist: str = "hamming") -> None:
        """
        Create a Permutation instance

        :param number_of_tiles: int,  number of permutation items. Needs to be bigger than zero and smaller than ten. For bigger
        values, the permutation set becomes to big.
        :param number_of_permutations: int, How many permutations to select
        :param criterion: callable The criterion by which the permutations are selected
        :param seed: int, Seed for the random module
        :param filename: string, A file to read the permutations from
        """

        if number_of_tiles < 1 or number_of_tiles > 9:
            raise ValueError(
                "The value for the permutation length needs to be bigger than zero and smaller than ten. number_of_tiles was {}"
                    .format(number_of_tiles))

        if not callable(criterion):
            raise ValueError("The given criterion was not a function.")

        if dist in ['d', 'hamming']:
            self.distance_to_use = dist
        else:
            raise ValueError('Input distance unknown')

        self.number_of_tiles = number_of_tiles
        self.__identity_permutation = tuple(range(number_of_tiles))
        self.criterion = criterion
        self.number_of_permutations = number_of_permutations

        if filename is None:
            self.permutations = self.__generate_custom_hamming_distance_set(number_of_permutations, criterion)
        else:
            self.read_permutations_from_file(filename)

        self.__seed = seed
        if seed is not None:
            random.seed(self.__seed)



    def __str__(self):
        result = ""
        for p in self.permutations:
            result += "{index}: \t\t{permutation}\n".format(**p)
        return result

    def __repr__(self):
        return "Permutator(\nn: \t\t{}\n".format(self.number_of_tiles) + "criterion: \t\t{}\n".format(
            self.criterion) + self.__str__() + ")"

    def read_permutations_from_file(self, filename):

        if type(filename) is str:
            if not os.path.isfile(filename):
                raise ValueError('The given permutation file did not exist.')

            with open(filename, 'r') as file:
                self.permutations = self.__read_permutations(file)
        elif issubclass(type(filename), pathlib.Path):
            if not filename.is_file():
                raise ValueError('The given permutation file did not exist.')

            with filename.open() as file:
                self.permutations = self.__read_permutations(file)
        else:
            raise ValueError('The given filename was not a string and not a pathlib path.')

    @staticmethod
    def __read_permutations(file):
        permutations = []

        reader = csv.reader(file, delimiter=',')

        # Skip the header line
        next(reader, None)

        i = 0
        for row in reader:
            permutations.append({'permutation': list(map(int, row[1:])), 'index': i})
            i = i + 1
        return permutations

    def save_permutations_to_file(self, filename):
        """
        Save this Permutators permutations to a .csv file

        :param filename: The filename of the file to save in
        :return: None
        """
        if not type(filename) is str:
            raise ValueError('The given filename was not a string.')

        with open(filename, 'w') as file:
            writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE)

            # Write title row
            writer.writerow(['Index', 'perm0', 'perm1', 'perm2', 'perm3', 'perm4', 'perm5', 'perm6', 'perm7', 'perm8'])

            # Write permutations.
            # Each entry consists of
            # index_of_permutation first_permutation_parameter second_permutation_parameter ...

            for perm in self.permutations:
                writer.writerow([str(perm['index'])] + list(perm['permutation']))

    def randomly_choose_permutation(self):
        """
        Randomly choose a permutation from the pre-generated list of permutations.

        :return: A randomly chosen permutation from the pre-generated permutation list.
        """
        return random.choice(self.permutations)

    def regenerate_permutations(self):
        """
        Regenerate the list of selected permutations.
        """
        self.__generate_custom_hamming_distance_set(self.number_of_permutations, self.criterion)

    def calculate_statistics(self):
        """
        Calculate statistics about the internal list of selected permutations.
        This function calculates:
            - The minimum Hamming distance (to the identity) of all chosen permutations.
            - The maximum Hamming distance (to the identity) of all chosen permutations.
            - The mean Hamming distance (to the identity) of all chosen permutations.
            - The mean intra-list Hamming distance. This done by calculating the mean of each elements Hamming
            distance to all other elements for each element and then taking the mean over all these means.
        :return: The minimum, maximum and mean Hamming distance in the permutation list as well as the mean
        intra-list distance.
        """

        # First
        distances = [self.distance(p['permutation']) for p in self.permutations]
        min_h_dist = min(distances)
        max_h_dist = max(distances)
        mean_h_dist = np.mean(distances)

        def intra_list_dist(p1): return np.mean(
            [self.hamming_distance(p1, p2['permutation']) for p2 in self.permutations])

        intra_list_dist_mean = np.mean([intra_list_dist(p['permutation']) for p in self.permutations])

        return min_h_dist, max_h_dist, mean_h_dist, intra_list_dist_mean

    def __generate_custom_hamming_distance_set(self, number_of_permutations, criterion):
        """
        Generate a set of permutations with hamming distances maximised by a given criterion.
        The default is maximizing for maximum hamming distance.

        This function implements algorithm 1 from the paper 'Unsupervised Learning of
        Visual Representations by Solving Jigsaw Puzzles' by Noroozi and Favaro


        :param criterion: The criterion by which the permutations are selected.
        :param number_of_permutations: How many permutations to select.
        :return: A set containing number_of_permutations permutations selected by the given criterion.
        """
        if number_of_permutations > math.factorial(self.number_of_tiles):
            raise ValueError(
                "The choosen number of permutations was higher than the number of distinct permutations "
                "available.")

        if number_of_permutations < 1:
            raise ValueError("The number of desired permutations was too low. It was {} < 1".format(
                number_of_permutations))

        # Generate all possible permutations
        all_permutations = list(itertools.permutations(self.__identity_permutation))

        # Reverse the list since items with big hamming distance are more likely to appear near the end of the list
        # all_permutations.reverse()

        # Get the hamming distance for all possible permutations
        D = [self.distance(p) for p in all_permutations]

        # Initialize an empty list that will hold the permutations we return
        P = []

        # The first permutation to append is chosen randomly
        j = np.random.randint(0, math.factorial(self.number_of_tiles))
        i = 1

        # Now, greedily add permutations that have a maximal value according to the criterion,
        # until we have collected enough elements.
        while i < number_of_permutations + 1:
            # Append the last chosen permutation and delete if from both reference sets
            P.append(all_permutations[j])
            del all_permutations[j]
            del D[j]

            # Choose a new permutation and increment the counter
            c = criterion(D)
            j = random.choice([m for m, n in enumerate(D) if n == c])
            i += 1

        return list(map(lambda x: {'index': x[0], 'permutation': x[1]}, zip(range(len(P)), P)))

    def distance(self, p):
        """
        Calculates the hamming distance to the identity permutation (1, 2, 3, ... number_of_tiles).
        :param p: A permutation. Tuple
        :return: The hamming distance to the identity permutation
        """
        if len(p) != self.number_of_tiles:
            raise ValueError(
                "The length of the permutation was not correct. Correct length is {}".format(self.number_of_tiles))

        if self.distance_to_use == 'd':
            return self.d_distance_permutation(p)
        else:
            return self.hamming_distance(p, self.__identity_permutation)

    @staticmethod
    def __f(x):
        x_dec = x - 1
        return np.asarray((math.floor(x_dec / 3), x_dec % 3))

    def d_distance_item(self, a, b):
        return np.abs(self.__f(a) - self.__f(b)).sum()

    def d_distance_permutation(self, p: np.ndarray) -> int:
        if len(p) != len(self.__identity_permutation):
            raise ValueError("Permutations had different shapes.")
        result: int = 0
        for a, b in zip(p, self.__identity_permutation):
            result += self.d_distance_item(a, b)
        return result

    @staticmethod
    def hamming_distance(p1, p2):
        """
        Calculates the hamming distance of two permutations. The hamming distance is defined as the number of
        elements in the permutation that are equal with respect to position and value. :param p1: First permutation. (Tuple)
        :param p1: First permutation
        :param p2: Second permutation.
        :return: The hamming distance of the two permutations.
        """
        if len(p1) != len(p2):
            raise ValueError('Hamming distance is undefined for tuples of unequal length.')

        return sum(a != b for (a, b) in zip(p1, p2))

    @staticmethod
    def inverse_permutation(p):
        """
        Get the inverse of a permutation
        :param p: Permutation, dict or list
        :return: List, inverse of the permutation
        """
        if isinstance(p, dict):
            p = p['permutation']

        inverse = [0] * len(p)
        for i, p in enumerate(p):
            inverse[p] = i
        return inverse

    def reorder_tiles(self, tiles):
        """
        Reorder tiles that were permutated.
        :param tiles: A dict, with fields 'tiles' and 'indices'.
        :return: A tensor containing the reordered tiles
        """
        inverse = self.inverse_permutation(self.permutations[tiles['indices']])
        return tiles['tiles'][inverse, :, :, :]
