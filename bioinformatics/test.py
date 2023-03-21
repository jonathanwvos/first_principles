from utils import (
    approx_pattern_count,
    approx_pattern_matching,
    frequent_words,
    hamming_distance,
    minimum_skew,
    pattern_matching,
    reverse_complement,
    skew_array,
    symbol_array
)
from unittest import TestCase, main


class TestBioinformatics(TestCase):
    '''The testing class for all bioinformatics related tests.'''

    def test_frequent_words(self):
        test_cases = [
            ['ACGTTGCATGTCGCATGATGCATGAGAGCT', 4, ['CATG', 'GCAT']],
            ['TGGTAGCGACGTTGGTCCCGCCGCTTGAGAATCTGGATGAACATAAGCTCCCACTTGGCTTATTCAGAGAACTGGTCAACACTTGTCTCTCCCAGCCAGGTCTGACCACCGGGCAACTTTTAGAGCACTATCGTGGTACAAATAATGCTGCCAC', 3, ['TGG']],
            ['CAGTGGCAGATGACATTTTGCTGGTCGACTGGTTACAACAACGCCTGGGGCTTTTGAGCAACGAGACTTTTCAATGTTGCACCGTTTGCTGCATGATATTGAAAACAATATCACCAAATAAATAACGCCTTAGTAAGTAGCTTTT', 4, ['TTTT']],
            ['ATACAATTACAGTCTGGAACCGGATGAACTGGCCGCAGGTTAACAACAGAGTTGCCAGGCACTGCCGCTGACCAGCAACAACAACAATGACTTTGACGCGAAGGGGATGGCATGAGCGAACTGATCGTCAGCCGTCAGCAACGAGTATTGTTGCTGACCCTTAACAATCCCGCCGCACGTAATGCGCTAACTAATGCCCTGCTG', 5, ['AACAA']],
            ['CCAGCGGGGGTTGATGCTCTGGGGGTCACAAGATTGCATTTTTATGGGGTTGCAAAAATGTTTTTTACGGCAGATTCATTTAAAATGCCCACTGGCTGGAGACATAGCCCGGATGCGCGTCTTTTACAACGTATTGCGGGGTAAAATCGTAGATGTTTTAAAATAGGCGTAAC', 5, ['AAAAT', 'GGGGT', 'TTTTA']]
        ]

        for genome, word_len, test_words in test_cases:
            words = set(frequent_words(genome, word_len))
            test_words = set(test_words)

            self.assertTrue(test_words.issubset(words))
            self.assertTrue(words.issubset(test_words))


    def test_symbol_array(self):
        test_cases = [
            ['AAAAGGGG', 'A', {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 1, 6: 2, 7: 3}],
            ['AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', 'CC', {0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7, 7: 7, 8: 7, 9: 7, 10: 7, 11: 7, 12: 7, 13: 7, 14: 7, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7, 25: 7, 26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7, 34: 7, 35: 7, 36: 7, 37: 7, 38: 7, 39: 7, 40: 7, 41: 7, 42: 7, 43: 7, 44: 7, 45: 7, 46: 7, 47: 7, 48: 7, 49: 7, 50: 7, 51: 7, 52: 7, 53: 7, 54: 7, 55: 7, 56: 7, 57: 7, 58: 7, 59: 7, 60: 7, 61: 7, 62: 7, 63: 7, 64: 7, 65: 7, 66: 7, 67: 7, 68: 7, 69: 7, 70: 7, 71: 7, 72: 7, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7, 79: 7, 80: 7, 81: 7, 82: 7, 83: 7, 84: 7, 85: 7, 86: 7, 87: 7, 88: 7, 89: 7, 90: 7, 91: 7, 92: 7, 93: 7, 94: 7, 95: 7, 96: 7, 97: 7, 98: 7, 99: 7, 100: 7, 101: 7, 102: 7, 103: 7, 104: 7, 105: 7, 106: 7, 107: 7, 108: 7, 109: 7, 110: 7, 111: 7, 112: 7, 113: 7, 114: 7, 115: 7, 116: 7, 117: 7, 118: 7, 119: 7, 120: 7, 121: 7, 122: 7, 123: 7, 124: 7, 125: 7, 126: 7, 127: 7, 128: 7, 129: 7, 130: 7, 131: 7, 132: 7, 133: 7, 134: 7}]
        ]

        for seq, symbol, result in test_cases:
            self.assertEqual(symbol_array(seq, symbol), result)


    def test_skew_array(self):
        test_cases = [
            ['CATGGGCATCGGCCATACGCC', [0, -1, -1, -1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0, -1, 0, -1, -2]],
            ['AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', [0, 0, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 0, -1, -1, 0, 0, -1, -2, -2, -1, -2, -2, -1, -2, -1, 0, 0, 1, 2, 1, 0, 0, -1, 0, -1, -2, -1, -1, -2, -2, -2, -3, -3, -4, -3, -2, -2, -2, -1, -2, -3, -3, -3, -2, -2, -1, -2, -2, -2, -2, -1, -1, 0, 1, 1, 1, 2, 1, 2, 2, 3, 2, 2, 2, 2, 3, 4, 4, 5, 6, 6, 6, 6, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 2, 1, 0, 1, 1, 0, 0, 0, 0]]
        ]

        for seq, test_skew in test_cases:
            self.assertEqual(skew_array(seq), test_skew)


    def test_minimum_skew(self):
        test_cases = [
            ['TAAAGACTGCCGAGAGGCCAACACGAGTGCTAGAACGAGGGGCGTAAACGCGGGTCCGAT', [11, 24]],
            ['ACCG', [3]],
            ['ACCC', [4]],
            ['CCGGGT', [2]],
            ['CCGGCCGG', [2, 6]],
            ['AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', [52]]
        ]

        for seq, test_mins in test_cases:
            self.assertEqual(minimum_skew(seq), test_mins)


    def test_reverse_complement(self):
        test_cases = [
            ['AAAACCCGGT', 'ACCGGGTTTT'],
            ['ACACAC', 'GTGTGT'],
            ['AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC', 'GCTGCTATCAGACACTCTTTTTTTAATCCACACAGAGACATATTGCCCGTTGCAGTCAGAATGAAAAGCT']
        ]

        for orig_genome, rev_com_genome in test_cases:
            self.assertEqual(reverse_complement(orig_genome), rev_com_genome)


    def test_pattern_matching(self):
        test_cases = [
            ['ATAT', 'GATATATGCATATACTT', [1, 3, 9]],
            ['ACAC', 'TTTTACACTTTTTTGTGTAAAAA', [4]],
            ['AAA', 'AAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATAGCGCACAGACAGATAATAATTACAGAGTACACAACATCCAT', [0, 46, 51, 74]],
            ['TTT', 'AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', [88, 92, 98, 132]],
            ['ATA', 'ATATATA', [0, 2, 4]]
        ]

        for pattern, genome, result in test_cases:
            self.assertEqual(pattern_matching(pattern, genome), result)


    def test_hamming_distance(self):
        test_cases = [
            ['GGGCCGTTGGT', 'GGACCGTTGAC', 3],
            ['AAAA', 'TTTT', 4],
            ['ACGTACGT', 'TACGTACG', 8],
            ['ACGTACGT', 'CCCCCCCC', 6],
            ['ACGTACGT', 'TGCATGCA', 8],
            ['GATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACT', 'AATAGCAGCTTCTCAACTGGTTACCTCGTATGAGTAAATTAGGTCATTATTGACTCAGGTCACTAACGTCT', 15],
            ['AGAAACAGACCGCTATGTTCAACGATTTGTTTTATCTCGTCACCGGGATATTGCGGCCACTCATCGGTCAGTTGATTACGCAGGGCGTAAATCGCCAGAATCAGGCTG', 'AGAAACCCACCGCTAAAAACAACGATTTGCGTAGTCAGGTCACCGGGATATTGCGGCCACTAAGGCCTTGGATGATTACGCAGAACGTATTGACCCAGAATCAGGCTC', 28],
            ['AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', 'AGAAACAGACCGCTATGTTCAACGATTTGTTTTATCTCGTCACCGGGATATTGCGGCCACTCATCGGTCAGTTGATTACGCAGGGCGTAAATCGCCAGAATCAGGCTGAAACCCTACGGACAGGTTTACGAACCT', 103]
        ]

        for a, b, d in test_cases:
            self.assertEqual(hamming_distance(a, b), d)


    def test_approx_pattern_matching(self):
        test_cases = [
            ['ATTCTGGA', 'CGCCCGAATCCAGAACGCATTCCCATATTTCGGGACCACTGGCCTCCACGGTACGGACGTCAATCAAAT', 3, [6, 7, 26, 27]],
            ['AAA', 'TTTTTTAAATTTTAAATTTTTT', 2, [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]],
            ['GAGCGCTGG', 'GAGCGCTGGGTTAACTCGCTACTTCCCGACGAGCGCTGTGGCGCAAATTGGCGATGAAACTGCAGAGAGAACTGGTCATCCAACTGAATTCTCCCCGCTATCGCATTTTGATGCGCGCCGCGTCGATT', 2, [0, 30, 66]],
            ['AATCCTTTCA', 'CCAAATCCCCTCATGGCATGCATTCCCGCAGTATTTAATCCTTTCATTCTGCATATAAGTAGTGAAGGTATAGAAACCCGTTCAAGCCCGCAGCGGTAAAACCGAGAACCATGATGAATGCACGGCGATTGCGCCATAATCCAAACA', 3, [3, 36, 74, 137]],
            ['CCGTCATCC', 'CCGTCATCCGTCATCCTCGCCACGTTGGCATGCATTCCGTCATCCCGTCAGGCATACTTCTGCATATAAGTACAAACATCCGTCATGTCAAAGGGAGCCCGCAGCGGTAAAACCGAGAACCATGATGAATGCACGGCGATTGC', 3, [0, 7, 36, 44, 48, 72, 79, 112]],
            ['TTT', 'AAAAAA', 3, [0, 1, 2, 3]],
            ['CCA', 'CCACCT', 0, [0]],
            ['GTGCCG', 'AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', 3, [3, 13, 16, 22, 25, 27, 28, 30, 33, 34, 36, 39, 47, 54, 61, 71, 76, 84, 87, 91, 101, 106, 119, 124]]
        ]

        for pattern, seq, ham_dist, test_positions in test_cases:
            self.assertEqual(approx_pattern_matching(seq, pattern, ham_dist), test_positions)


    def test_approx_pattern_count(self):
        test_cases = [
            ['GAGG', 'TTTAGAGCCTTCAGAGG', 2, 4],
            ['AA', 'AAA', 0, 2],
            ['ATA', 'ATA', 1, 1],
            ['GTGCCG', 'AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT', 3, 24]
        ]

        for pattern, seq, ham_dist, test_count in test_cases:
            self.assertEqual(approx_pattern_count(seq, pattern, ham_dist), test_count)


if __name__ == '__main__':
    main()
