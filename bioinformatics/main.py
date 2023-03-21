from utils import skew_array, minimum_skew
from time import time


if __name__ == '__main__':
    genome = 'CATGGGCATCGGCCATACGCC'
    test_res = [0, -1, -1, -1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0, -1, 0, -1, -2]

    print(minimum_skew(genome))