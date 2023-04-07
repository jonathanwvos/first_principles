from utils import (
    approx_pattern_count,
    approx_pattern_matching,
    consensus_motif,
    count_motifs,
    frequent_words,
    greedy_motif_search,
    hamming_distance,
    minimum_skew,
    motifs,
    normalize,
    profiles,
    pattern_count,
    pattern_matching,
    profile_most_probable_kmer,
    profile_prob,
    pseudo_count_motifs,
    pseudo_greedy_motif_search,
    pseudo_profiles,
    reverse_complement,
    score_consensus_motif,
    skew_array,
    symbol_array,
    weighted_die
)
from unittest import TestCase, main

import numpy as np


class TestBioinformatics(TestCase):
    '''The testing class for all bioinformatics related tests.'''

    def test_pattern_count(self):
        test_cases = [
            [
                'GCGCG',
                'GCG',
                2
            ],[
                'ACGTACGTACGT',
                'CG',
                3
            ],[
                'AAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATAGCGCACAGACAGATAATAATTACAGAGTACACAACATCCAT',
                'AAA',
                4
            ],[
                'AGCGTGCCGAAATATGCCGCCAGACCTGCTGCGGTGGCCTCGCCGACTTCACGGATGCCAAGTGCATAGAGGAAGCGAGCAAAGGTGGTTTCTTTCGCTTTATCCAGCGCGTTAACCACGTTCTGTGCCGACTTT',
                'TTT',
                4
            ],[
                'GGACTTACTGACGTACG',
                'ACT',
                2
            ],[
                'ATCCGATCCCATGCCCATG',
                'CC',
                5
            ],[
                'CTGTTTTTGATCCATGATATGTTATCTCTCCGTCATCAGAAGAACAGTGACGGATCGCCCTCTCTCTTGGTCAGGCGACCGTTTGCCATAATGCCCATGCTTTCCAGCCAGCTCTCAAACTCCGGTGACTCGCGCAGGTTGAGTA',
                'CTC',
                9
            ]
        ]

        for genome, pattern, count in test_cases:
            self.assertEqual(pattern_count(pattern, genome), count)

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

        exception_case = test_cases[0]
        exception_case[0] = exception_case[0:-2]
        with self.assertRaises(Exception):
            hamming_distance(exception_case[0], exception_case[1])

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

    def test_count_motifs(self):
        test_cases = [
            [['AACGTA', 'CCCGTT', 'CACCTT', 'GGATTA', 'TTCCGG'], {'A': np.array([1, 2, 1, 0, 0, 2]), 'C': np.array([2, 1, 4, 2, 0, 0]), 'G': np.array([1, 1, 0, 2, 1, 1]), 'T': np.array([1, 1, 0, 1, 4, 2])}],
            [['GTACAACTGT', 'CAACTATGAA', 'TCCTACAGGA', 'AAGCAAGGGT', 'GCGTACGACC', 'TCGTCAGCGT', 'AACAAGGTCA', 'CTCAGGCGTC', 'GGATCCAGGT', 'GGCAAGTACC'], {'A': np.array([2, 3, 3, 3, 6, 4, 2, 2, 1, 3]), 'C': np.array([2, 3, 4, 3, 2, 3, 2, 1, 3, 3]), 'G': np.array([4, 2, 3, 0, 1, 3, 4, 5, 5, 0]), 'T': np.array([2, 2, 0, 4, 1, 0, 2, 2, 1, 4])}]
        ]

        with self.assertRaises(Exception):
            count_motifs([])

        with self.assertRaises(Exception):
            count_motifs([''])

        for motifs, exp_results in test_cases:
            results = count_motifs(motifs)

            for n in ['A', 'C', 'G', 'T']:
                self.assertTrue(np.array_equal(results[n], exp_results[n]))

    def test_profiles(self):
        test_cases = [
            [['AACGTA', 'CCCGTT', 'CACCTT', 'GGATTA', 'TTCCGG'], {'A': np.array([1, 2, 1, 0, 0, 2])/5, 'C': np.array([2, 1, 4, 2, 0, 0])/5, 'G': np.array([1, 1, 0, 2, 1, 1])/5, 'T': np.array([1, 1, 0, 1, 4, 2])/5}],
            [['GTACAACTGT', 'CAACTATGAA', 'TCCTACAGGA', 'AAGCAAGGGT', 'GCGTACGACC', 'TCGTCAGCGT', 'AACAAGGTCA', 'CTCAGGCGTC', 'GGATCCAGGT', 'GGCAAGTACC'], {'A': np.array([2, 3, 3, 3, 6, 4, 2, 2, 1, 3])/10, 'C': np.array([2, 3, 4, 3, 2, 3, 2, 1, 3, 3])/10, 'G': np.array([4, 2, 3, 0, 1, 3, 4, 5, 5, 0])/10, 'T': np.array([2, 2, 0, 4, 1, 0, 2, 2, 1, 4])/10}]
        ]

        with self.assertRaises(Exception):
            profiles([])

        with self.assertRaises(Exception):
            profiles([''])

        for profile, exp_results in test_cases:
            results = profiles(profile)

            for n in ['A', 'C', 'G', 'T']:
                self.assertTrue(np.array_equal(results[n], exp_results[n]))

    def test_consensus_motifs(self):
        test_cases = [
            [['AACGTA', 'CCCGTT', 'CACCTT', 'GGATTA', 'TTCCGG'], 'CACCTA'],
            [['GTACAACTGT', 'CAACTATGAA', 'TCCTACAGGA', 'AAGCAAGGGT', 'GCGTACGACC', 'TCGTCAGCGT', 'AACAAGGTCA', 'CTCAGGCGTC', 'GGATCCAGGT', 'GGCAAGTACC'], 'GACTAAGGGT']
        ]

        for motifs, test_consensus in test_cases:
            self.assertTrue(hamming_distance(consensus_motif(motifs), test_consensus) <= 2)

    def test_score_consensus_motif(self):
        test_cases = [
            [['AACGTA', 'CCCGTT', 'CACCTT', 'GGATTA', 'TTCCGG'], 14],
            [['GTACAACTGT', 'CAACTATGAA', 'TCCTACAGGA', 'AAGCAAGGGT', 'GCGTACGACC', 'TCGTCAGCGT', 'AACAAGGTCA', 'CTCAGGCGTC', 'GGATCCAGGT', 'GGCAAGTACC'], 57]
        ]

        for motifs, test_score in test_cases:
            self.assertEqual(score_consensus_motif(motifs), test_score)

    def test_profile_prob(self):
        test_cases = [
            ['ACGGGGATTACC',
            {'A': [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.1, 0.1, 0.3, 0.0],
            'C': [0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.2, 0.4, 0.6],
            'G': [0.0, 0.0, 1.0, 1.0, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            'T': [0.7, 0.2, 0.0, 0.0, 0.1, 0.1, 0.0, 0.5, 0.8, 0.7, 0.3, 0.4]},
            0.000839808],
            ['TCGGGGGCCACC',
            {'A': [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.1, 0.1, 0.3, 0.0],
            'C': [0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.2, 0.4, 0.6],
            'G': [0.0, 0.0, 1.0, 1.0, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            'T': [0.7, 0.2, 0.0, 0.0, 0.1, 0.1, 0.0, 0.5, 0.8, 0.7, 0.3, 0.4]},
            3.26592e-05]
        ]

        for profile, test_scores, test_prob in test_cases:
            self.assertAlmostEqual(profile_prob(profile, test_scores), test_prob, 9)

    def test_profile_most_probable_kmer(self):
        test_cases = [
            [
                'ACCTGTTTATTGCCTAAGTTCCGAACAAACCCAATATAGCCCGAGGGCCT',
                5,
                {'A': [0.2, 0.2, 0.3, 0.2, 0.3],
                'C': [0.4, 0.3, 0.1, 0.5, 0.1],
                'G': [0.3, 0.3, 0.5, 0.2, 0.4],
                'T': [0.1, 0.2, 0.1, 0.1, 0.2]},
                'CCGAG'
            ],
            [
                'AGCAGCTTTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATCTGAACTGGTTACCTGCCGTGAGTAAAT',
                8,
                {'A': [0.7, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1],
                'C': [0.2, 0.2, 0.5, 0.4, 0.2, 0.3, 0.1, 0.6],
                'G': [0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2],
                'T': [0.0, 0.3, 0.2, 0.0, 0.2, 0.3, 0.3, 0.1]},
                'AGCAGCTT'
            ],
            [
                'TTACCATGGGACCGCTGACTGATTTCTGGCGTCAGCGTGATGCTGGTGTGGATGACATTCCGGTGCGCTTTGTAAGCAGAGTTTA',
                12,
                {'A': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5],
                'C': [0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.4, 0.3, 0.2, 0.2, 0.1],
                'G': [0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.1],
                'T': [0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.0, 0.2, 0.4, 0.4, 0.2, 0.3]},
                'AAGCAGAGTTTA'
            ],
            [
                'AACCGGTT',
                3,
                {'A': [1.0, 1.0, 1.0],
                'C': [0.0, 0.0, 0.0],
                'G': [0.0, 0.0, 0.0],
                'T': [0.0, 0.0, 0.0]},
                'AAC'
            ],
            [
                'TTACCATGGGACCGCTGACTGATTTCTGGCGTCAGCGTGATGCTGGTGTGGATGACATTCCGGTGCGCTTTGTAAGCAGAGTTTA',
                5,
                {'A': [0.2, 0.2, 0.3, 0.2, 0.3],
                'C': [0.4, 0.3, 0.1, 0.5, 0.1],
                'G': [0.3, 0.3, 0.5, 0.2, 0.4],
                'T': [0.1, 0.2, 0.1, 0.1, 0.2]},
                'CAGCG'
            ]
        ]

        for seq, k, profile_scores, kmer in test_cases:
            self.assertEqual(profile_most_probable_kmer(seq, k, profile_scores), kmer)

    def test_greedy_motif_search(self):
        test_cases = [
            [
                3, 5, 
                ['GGCGTTCAGGCA',
                'AAGAATCAGTCA',
                'CAAGGAGTTCGC',
                'CACGTCAATCAC',
                'CAATAATATTCG'],
                ['CAG',
                'CAG',
                'CAA',
                'CAA',
                'CAA']
            ],
            [
                3, 4,
                ['GCCCAA',
                'GGCCTG',
                'AACCTA',
                'TTCCTT'],
                ['GCC',
                'GCC',
                'AAC',
                'TTC']
            ],
            [
                5, 8,
                ['GAGGCGCACATCATTATCGATAACGATTCGCCGCATTGCC',
                'TCATCGAATCCGATAACTGACACCTGCTCTGGCACCGCTC',
                'TCGGCGGTATAGCCAGAAAGCGTAGTGCCAATAATTTCCT',
                'GAGTCGTGGTGAAGTGTGGGTTATGGGGAAAGGCAGACTG',
                'GACGGCAACTACGGTTACAACGCAGCAACCGAAGAATATT',
                'TCTGTTGTTGCTAACACCGTTAAAGGCGGCGACGGCAACT',
                'AAGCGGCCAACGTAGGCGCGGCTTGGCATCTCGGTGTGTG',
                'AATTGAAAGGCGCATCTTACTCTTTTCGCTTTCAAAAAAA'],
                ['GAGGC',
                'TCATC',
                'TCGGC',
                'GAGTC',
                'GCAGC',
                'GCGGC',
                'GCGGC',
                'GCATC']
            ],
            [
                6, 5,
                ['GCAGGTTAATACCGCGGATCAGCTGAGAAACCGGAATGTGCGT',
                'CCTGCATGCCCGGTTTGAGGAACATCAGCGAAGAACTGTGCGT',
                'GCGCCAGTAACCCGTGCCAGTCAGGTTAATGGCAGTAACATTT',
                'AACCCGTGCCAGTCAGGTTAATGGCAGTAACATTTATGCCTTC',
                'ATGCCTTCCGCGCCAATTGTTCGTATCGTCGCCACTTCGAGTG'],
                ['GTGCGT',
                'GTGCGT',
                'GCGCCA',
                'GTGCCA',
                'GCGCCA']
            ],
            [
                5, 8,
                ['GACCTACGGTTACAACGCAGCAACCGAAGAATATTGGCAA',
                'TCATTATCGATAACGATTCGCCGGAGGCCATTGCCGCACA',
                'GGAGTCTGGTGAAGTGTGGGTTATGGGGCAGACTGGGAAA',
                'GAATCCGATAACTGACACCTGCTCTGGCACCGCTCTCATC',
                'AAGCGCGTAGGCGCGGCTTGGCATCTCGGTGTGTGGCCAA',
                'AATTGAAAGGCGCATCTTACTCTTTTCGCTTAAAATCAAA',
                'GGTATAGCCAGAAAGCGTAGTTAATTTCGGCTCCTGCCAA',
                'TCTGTTGTTGCTAACACCGTTAAAGGCGGCGACGGCAACT'],
                ['GCAGC',
                'TCATT',
                'GGAGT',
                'TCATC',
                'GCATC',
                'GCATC',
                'GGTAT',
                'GCAAC']
            ],
            [
                4, 8,
                ['GACCTACGGTTACAACGCAGCAACCGAAGAATATTGGCAA',
                'TCATTATCGATAACGATTCGCCGGAGGCCATTGCCGCACA',
                'GGAGTCTGGTGAAGTGTGGGTTATGGGGCAGACTGGGAAA',
                'GAATCCGATAACTGACACCTGCTCTGGCACCGCTCTCATC',
                'AAGCGCGTAGGCGCGGCTTGGCATCTCGGTGTGTGGCCAA',
                'AATTGAAAGGCGCATCTTACTCTTTTCGCTTAAAATCAAA',
                'GGTATAGCCAGAAAGCGTAGTTAATTTCGGCTCCTGCCAA',
                'TCTGTTGTTGCTAACACCGTTAAAGGCGGCGACGGCAACT'],
                ['CGCA',
                'CGCA',
                'GGAG',
                'GGCA',
                'GGCA',
                'CGCA',
                'GGTA',
                'GGCA']
            ]
        ]

        for k, t, dna, test_motifs in test_cases:
            self.assertEqual(greedy_motif_search(dna, k, t), test_motifs)

    def test_pseudo_counts(self):
        test_cases = [
            [
                ['AACGTA',
                'CCCGTT',
                'CACCTT',
                'GGATTA',
                'TTCCGG'],
                {'A': [2, 3, 2, 1, 1, 3], 'C': [3, 2, 5, 3, 1, 1], 'T': [2, 2, 1, 2, 5, 3], 'G': [2, 2, 1, 3, 2, 2]}
            ],
            [
                ['GTACAACTGT',
                'CAACTATGAA',
                'TCCTACAGGA',
                'AAGCAAGGGT',
                'GCGTACGACC',
                'TCGTCAGCGT',
                'AACAAGGTCA',
                'CTCAGGCGTC',
                'GGATCCAGGT',
                'GGCAAGTACC'],
                {'A': [3, 4, 4, 4, 7, 5, 3, 3, 2, 4], 'C': [3, 4, 5, 4, 3, 4, 3, 2, 4, 4], 'T': [3, 3, 1, 5, 2, 1, 3, 3, 2, 5], 'G': [5, 3, 4, 1, 2, 4, 5, 6, 6, 1]}
            ]
        ]

        for motifs, test_results in test_cases:
            self.assertEqual(pseudo_count_motifs(motifs), test_results)

    def test_pseudo_profiles(self):
        test_cases = [
            [
                ['AACGTA',
                'CCCGTT',
                'CACCTT',
                'GGATTA',
                'TTCCGG'],
                {'A': [0.2222222222222222, 0.3333333333333333, 0.2222222222222222, 0.1111111111111111, 0.1111111111111111, 0.3333333333333333], 'C': [0.3333333333333333, 0.2222222222222222, 0.5555555555555556, 0.3333333333333333, 0.1111111111111111, 0.1111111111111111], 'T': [0.2222222222222222, 0.2222222222222222, 0.1111111111111111, 0.2222222222222222, 0.5555555555555556, 0.3333333333333333], 'G': [0.2222222222222222, 0.2222222222222222, 0.1111111111111111, 0.3333333333333333, 0.2222222222222222, 0.2222222222222222]},
            ],
            [
                ['GTACAACTGT',
                'CAACTATGAA',
                'TCCTACAGGA',
                'AAGCAAGGGT',
                'GCGTACGACC',
                'TCGTCAGCGT',
                'AACAAGGTCA',
                'CTCAGGCGTC',
                'GGATCCAGGT',
                'GGCAAGTACC'],
                {'A': [0.21428571428571427, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.5, 0.35714285714285715, 0.21428571428571427, 0.21428571428571427, 0.14285714285714285, 0.2857142857142857], 'C': [0.21428571428571427, 0.2857142857142857, 0.35714285714285715, 0.2857142857142857, 0.21428571428571427, 0.2857142857142857, 0.21428571428571427, 0.14285714285714285, 0.2857142857142857, 0.2857142857142857], 'T': [0.21428571428571427, 0.21428571428571427, 0.07142857142857142, 0.35714285714285715, 0.14285714285714285, 0.07142857142857142, 0.21428571428571427, 0.21428571428571427, 0.14285714285714285, 0.35714285714285715], 'G': [0.35714285714285715, 0.21428571428571427, 0.2857142857142857, 0.07142857142857142, 0.14285714285714285, 0.2857142857142857, 0.35714285714285715, 0.42857142857142855, 0.42857142857142855, 0.07142857142857142]}
            ]
        ]

        for motifs, test_results in test_cases:
            self.assertEqual(pseudo_profiles(motifs), test_results)

    def test_pseudo_greedy_motif_search(self):
        test_cases = [
            [
                3, 5,
                [
                    'GGCGTTCAGGCA',
                    'AAGAATCAGTCA',
                    'CAAGGAGTTCGC',
                    'CACGTCAATCAC',
                    'CAATAATATTCG'
                ],
                [
                    'TTC',
                    'ATC',
                    'TTC',
                    'ATC',
                    'TTC'
                ]
            ],[
                5, 8,
                [
                    'AGGCGGCACATCATTATCGATAACGATTCGCCGCATTGCC',
                    'ATCCGTCATCGAATAACTGACACCTGCTCTGGCACCGCTC',
                    'AAGCGTCGGCGGTATAGCCAGATAGTGCCAATAATTTCCT',
                    'AGTCGGTGGTGAAGTGTGGGTTATGGGGAAAGGCAGACTG',
                    'AACCGGACGGCAACTACGGTTACAACGCAGCAAGAATATT',
                    'AGGCGTCTGTTGTTGCTAACACCGTTAAGCGACGGCAACT',
                    'AAGCGGCCAACGTAGGCGCGGCTTGGCATCTCGGTGTGTG',
                    'AATTGAAAGGCGCATCTTACTCTTTTCGCTTTCAAAAAAA'
                ],
                [
                    'AGGCG',
                    'ATCCG',
                    'AAGCG',
                    'AGTCG',
                    'AACCG',
                    'AGGCG',
                    'AGGCG',
                    'AGGCG'
                ]
            ],[
                5, 8,
                [
                    'GCACATCATTAAACGATTCGCCGCATTGCCTCGATAGGCG',
                    'TCATAACTGACACCTGCTCTGGCACCGCTCATCCGTCGAA',
                    'AAGCGGGTATAGCCAGATAGTGCCAATAATTTCCTTCGGC',
                    'AGTCGGTGGTGAAGTGTGGGTTATGGGGAAAGGCAGACTG',
                    'AACCGGACGGCAACTACGGTTACAACGCAGCAAGAATATT',
                    'AGGCGTCTGTTGTTGCTAACACCGTTAAGCGACGGCAACT',
                    'AAGCTTCCAACATCGTCTTGGCATCTCGGTGTGTGAGGCG',
                    'AATTGAACATCTTACTCTTTTCGCTTTCAAAAAAAAGGCG'
                ],
                [
                    'AGGCG',
                    'TGGCA',
                    'AAGCG',
                    'AGGCA',
                    'CGGCA',
                    'AGGCG',
                    'AGGCG',
                    'AGGCG'
                ]
            
            ],[
                5, 8,
                [
                    'GCACATCATTATCGATAACGATTCATTGCCAGGCGGCCGC',
                    'TCATCGAATAACTGACACCTGCTCTGGCTCATCCGACCGC',
                    'TCGGCGGTATAGCCAGATAGTGCCAATAATTTCCTAAGCG',
                    'GTGGTGAAGTGTGGGTTATGGGGAAAGGCAGACTGAGTCG',
                    'GACGGCAACTACGGTTACAACGCAGCAAGAATATTAACCG',
                    'TCTGTTGTTGCTAACACCGTTAAGCGACGGCAACTAGGCG',
                    'GCCAACGTAGGCGCGGCTTGGCATCTCGGTGTGTGAAGCG',
                    'AAAGGCGCATCTTACTCTTTTCGCTTTCAAAAAAAAATTG'
                ],
                [
                    'GGCGG',
                    'GGCTC',
                    'GGCGG',
                    'GGCAG',
                    'GACGG',
                    'GACGG',
                    'GGCGC',
                    'GGCGC'
                ]
            ],
            [
                3, 8,
                [
                    'GCACATCATTATCGATAACGATTCATTGCCAGGCGGCCGC',
                    'TCATCGAATAACTGACACCTGCTCTGGCTCATCCGACCGC',
                    'TCGGCGGTATAGCCAGATAGTGCCAATAATTTCCTAAGCG',
                    'GTGGTGAAGTGTGGGTTATGGGGAAAGGCAGACTGAGTCG',
                    'GACGGCAACTACGGTTACAACGCAGCAAGAATATTAACCG',
                    'TCTGTTGTTGCTAACACCGTTAAGCGACGGCAACTAGGCG',
                    'GCCAACGTAGGCGCGGCTTGGCATCTCGGTGTGTGAAGCG',
                    'AAAGGCGCATCTTACTCTTTTCGCTTTCAAAAAAAAATTG'
                ],
                [
                    'GGC',
                    'GGC',
                    'GGC',
                    'GGC',
                    'GGC',
                    'GGC',
                    'GGC',
                    'GGC'
                ]
            ]
        ]

        for k, t, dna, test_motifs in test_cases:
            self.assertEqual(pseudo_greedy_motif_search(dna, k, t), test_motifs)

    def test_motifs(self):
        test_cases = [
            [
                {
                    'A': [0.8, 0.0, 0.0, 0.2],
                    'C': [0.0, 0.6, 0.2, 0.0],
                    'G': [0.2, 0.2, 0.8, 0.0],
                    'T': [0.0, 0.2, 0.0, 0.8]
                },
                [
                    'TTACCTTAAC',
                    'GATGTCTGTC',
                    'ACGGCGTTAG',
                    'CCCTAACGAG',
                    'CGTCAGAGGT'
                ],
                [
                    'ACCT',
                    'ATGT',
                    'GCGT',
                    'ACGA',
                    'AGGT'
                ]
            ],
            [
                {
                    'A': [0.5, 0.0, 0.2, 0.2],
                    'C': [0.3, 0.6, 0.2, 0.0],
                    'G': [0.2, 0.2, 0.6, 0.0],
                    'T': [0.0, 0.2, 0.0, 0.8]
                },
                [
                    'TTACCTTAAC',
                    'GATGTCTGTC',
                    'ACGGCGTTAG',
                    'CCCTAACGAG',
                    'CGTCAGAGGT'
                ],
                [
                    'ACCT',
                    'ATGT',
                    'GCGT',
                    'ACGA',
                    'AGGT'
                ] 
            ]
        ]

        for profiles, dna, test_results in test_cases:
            self.assertEqual(motifs(profiles, 4, dna), test_results)

    def test_normalize(self):
        self.assertEqual(normalize({'A': 0.1, 'C': 0.1, 'G': 0.1, 'T': 0.1}), {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})

    def test_weighted_die(self):
        no_iter = 1000
        test_case = {
            'A': 0.25,
            'C': 0.25,
            'G': 0.25,
            'T': 0.25,
        }

        dist = dict.fromkeys(test_case.keys(), 0)

        for _ in range(no_iter):
            k_mer = weighted_die(test_case)

            dist[k_mer] += 1

        self.assertTrue(dist['A'] - no_iter//4 <= no_iter//10)
        self.assertTrue(dist['C'] - no_iter//4 <= no_iter//10)
        self.assertTrue(dist['G'] - no_iter//4 <= no_iter//10)
        self.assertTrue(dist['T'] - no_iter//4 <= no_iter//10)
        

if __name__ == '__main__':
    main()
