from utils import hamming_distance
from time import time


if __name__ == '__main__':
    genome = 'GATACACTTCCCGAGTAGGTACTG'
    test_res = [0, -1, -1, -1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0, -1, 0, -1, -2]

    print(hamming_distance('CTACAGCAATACGATCATATGCGGATCCGCAGTGGCCGGTAGACACACGT', 'CTACCCCGCTGCTCAATGACCGGGACTAAAGAGGCGAAGATTATGGTGTG'))

    