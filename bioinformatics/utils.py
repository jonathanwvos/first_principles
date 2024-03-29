from constants import AMINO_ACID_MAP, DNA_COMP_MAP, NUCLEOTIDES, RNA_COMP_MAP
from math import log2
from random import randint

import numpy as np

def count_nucleotides(dna: str) -> dict:
    '''Count the number of nucleotides in a dna sequence.'''
    nucleotides = dict.fromkeys(NUCLEOTIDES)
    dna = dna.upper()

    for n in nucleotides:
        nucleotides[n] = str(dna.count(n))

    return nucleotides


def dna_to_rna(dna: str) -> str:
    '''Transcribe dna to rna.'''
    dna = dna.upper()

    return dna.replace('T', 'U')


def complement(seq: str, type: str) -> str:
    '''Return the complement of the gene sequence.'''
    seq = seq.upper()

    comp = ''
    comp_map = DNA_COMP_MAP if type == 'dna' else RNA_COMP_MAP

    for i in range(len(seq)):
        comp += comp_map[seq[i]]
        
    return comp


def reverse(seq: str) -> str:
    '''Reverse Pattern.'''

    return seq[::-1]


def reverse_complement(seq: str) -> str:
    seq = seq.upper()

    if 'U' in seq:
        type = 'rna'
    else:
        type = 'dna'

    return reverse(complement(seq, type))


def frequent_words(text, k) -> list:
    '''Determine the most frequent subsequence of length k.'''

    words = []
    freq = frequency_map(text, k)
    m = max(freq.values())
    
    for key in freq:
        if freq[key] == m:
            words.append(key)

    return words


def frequency_map(text: str, k: int) -> dict:
    '''Determine the frequency of all words of length k.'''

    freq = {}
    n = len(text)
    for i in range(n-k+1):
        pattern = text[i:i+k]

        if pattern in freq:
            freq[pattern] += 1
        else:
            freq[pattern] = 0
    
    return freq


def pattern_matching(pattern: str, genome: str) -> list:
    '''Determine the starting positions of all matching patterns in the genome.'''
    
    positions = []
    pat_len = len(pattern)
    gen_len = len(genome)

    for i in range(0, gen_len-pat_len+1):
        if genome[i:i+pat_len] == pattern:
            positions.append(i)

    return positions


def pattern_count(pattern: str, genome: str):
    '''Count the number of matching patterns in the genome.'''
    
    return len(pattern_matching(pattern, genome))


def symbol_array(genome, symbol):
    '''
    Determine the number of symbols in the first half of the genome and increment/decrement
    for every cytosine/guanine molecule.
    '''

    array = {}
    n = len(genome)
    extended_genome = genome + genome[0:n//2]

    # look at the first half of Genome to compute first array value
    array[0] = pattern_count(symbol, genome[0:n//2])

    for i in range(1, n):
        # start by setting the current array value equal to the previous array value
        array[i] = array[i-1]

        # the current array value can differ from the previous array value by at most 1
        if extended_genome[i-1] == symbol:
            array[i] = array[i]-1
        if extended_genome[i+(n//2)-1] == symbol:
            array[i] = array[i]+1
    return array


def skew_array(genome: str) -> list:
    '''Determine the skew array of a genome by incrementing/decrementing by 1
    when a guanine/cytosine molecule is encountered.'''
    arr = [0]
    gen_len = len(genome)

    for i in range(gen_len):
        if genome[i] == 'G':
            arr.append(arr[-1]+1)
        elif genome[i] == 'C':
            arr.append(arr[-1]-1)
        else:
            arr.append(arr[-1])
    
    return arr


def minimum_skew(genome: str) -> list:
    '''Determine the positions of the minimum values in a skew array.'''

    positions = []
    skew = skew_array(genome)
    skew_len = len(skew)
    min_val = min(skew)

    for i in range(skew_len):
        if skew[i] == min_val:
            positions.append(i)

    return positions


def hamming_distance(seq_a: str, seq_b: str) -> int:
    '''Determine the hamming distance between two sequencs.'''

    if len(seq_a) != len(seq_b):
        raise Exception("Genomes A & B don't have the same length.")
    
    dist = 0

    for a,b in zip(seq_a, seq_b):
        if a != b:
            dist += 1

    return dist


def approx_pattern_matching(genome: str, pattern: str, ham_dist: int) -> list:
    '''
    Determine the starting positions of all matching patterns in the genome that are
    within ham_dist of each other.
    '''

    positions = []

    pat_len = len(pattern)
    gen_len = len(genome)

    for i in range(0, gen_len-pat_len+1):
        if hamming_distance(genome[i:i+pat_len], pattern) <= ham_dist:
            positions.append(i)

    return positions


def approx_pattern_count(genome: str, pattern: str, ham_dist: int) -> list:
    '''Count the number of approximate matching patterns in the genome.'''

    return len(approx_pattern_matching(genome, pattern, ham_dist))


def count_motifs(motifs: list) -> dict:
    '''Count the number of occurrences of each nucleotide in each colum
    of the motifs.'''

    if len(motifs) == 0 or len(motifs[0]) == 0:
        raise Exception('motifs args is empty.')

    motif_len = len(motifs[0])
    count = {
        'A': np.zeros(motif_len),
        'C': np.zeros(motif_len),
        'G': np.zeros(motif_len),
        'T': np.zeros(motif_len)
    }
    no_motifs = len(motifs)

    for i in range(no_motifs):
        for j in range(motif_len):
            symbol = motifs[i][j]
            count[symbol][j] += 1

    return count


def profiles(motifs):
    '''Determine the profiles of the motifs by normalizing nucleotides frequencies.'''

    if len(motifs) == 0 or len(motifs[0]) == 0:
        raise Exception('motifs args is empty.')
    
    no_motifs = len(motifs)
    profiles = count_motifs(motifs)

    for n in profiles:
        profiles[n] = profiles[n]/no_motifs

    return profiles


def consensus_motif(motifs):
    '''
    Determine the consensus motif by selecting the nucleotides with the highest probability using motif profiles.
    Ties are chosen randomly.
    '''

    mps = profiles(motifs)
    mp_arr = np.array(list(mps.values())).T
    con_motif = ''

    for m in mp_arr:
        max_profile = np.max(m)
        max_indices = np.where(m == max_profile)
        max_index = np.random.choice(max_indices[0], 1)
        con_motif += NUCLEOTIDES[max_index[0]]

    return con_motif


def score_consensus_motif(motifs):
    '''
    Determine the score for the consensus motifs by summing the total number
    of different nucleotides in the j-th column.
    '''

    score = 0
    con_motif = consensus_motif(motifs)
    motif_len = len(motifs[0])

    for i in range(motif_len):
        for motif in motifs:
            if motif[i] != con_motif[i]:
                score += 1

    return score


def profile_prob(profile, scores):
    '''
    Determine the probability of a k-mer being close to a consensus motif.
    
    Parameters
    ----------
    profile : str
        A candidate profile

    scores: dict
        A dictionary of scores of the form {'A': [], 'C': [], 'G': [], 'T': []}
    '''

    prob = 1.

    for idx, n in enumerate(profile):
        prob *= scores[n][idx]

    return prob


def profile_most_probable_kmer(seq, k, scores):
    '''
    Determine the profile-most probably k-mer in the sequence provided.
    
    Parameters
    ----------

    seq : str
        A string representing a genetic sequence or sub-sequence

    k : int
        The desired length of the k-mer to find

    scores: dict
        A dictionary of scores of the form {'A': [], 'C': [], 'G': [], 'T': []}
    '''

    seq_len = len(seq)
    most_prob_profile = ''
    highest_prob = -1.

    for i in range(0, seq_len - k + 1):
        profile = seq[i: i+k]
        prof_prob = profile_prob(profile, scores)

        if prof_prob > highest_prob:
            most_prob_profile = profile
            highest_prob = prof_prob

    return most_prob_profile


def greedy_motif_search(dna, k, t):
    '''Greedily determine the best rated motifs for a dna sequence.'''

    best_motifs = [d[0:k] for d in dna]
    dna_len = len(dna[0])

    for i in range(dna_len-k+1):
        motifs = [dna[0][i:i+k]]

        for j in range(1, t):
            profile = profiles(motifs[0:j])
            motifs.append(profile_most_probable_kmer(dna[j], k, profile))

        if score_consensus_motif(motifs) < score_consensus_motif(best_motifs):
            best_motifs = motifs

    return best_motifs


def translate_rna(mrna: str) -> str:
    '''Translate mRNA sequence to amino acid sequence.'''

    aa_seq = ''
    mrna_len = len(mrna)

    for i in range(0, mrna_len, 3):
        codon = mrna[i:i+3]
        aa_seq += AMINO_ACID_MAP[codon]

    return aa_seq


def dist_entropy(dist: list) -> float:
    '''Compute the entropy of the distribution provided.'''
    en = 0

    for p in dist:
        if p != 0:
            en += p*log2(p)

    return -en

def profile_entropy(profile: dict) -> float:
    '''
    Compute the entropy of the motif profile provided provided.
    
    profile: dict
        A dictionary of profile of the form {'A': [], 'C': [], 'G': [], 'T': []}
    '''

    dists = []
    for key in profile:
        dists.append(profile[key])

    dists = np.array(dists).T

    en = 0
    for dist in dists:
        en += dist_entropy(dist)

    return en


def pseudo_count_motifs(motifs: list) -> dict:
    '''
    Count the number of occurrences of each nucleotide in each colum
    of the motifs. Use Laplace's rule of Succession to prevent zero
    probabilities from arising.
    '''

    if len(motifs) == 0 or len(motifs[0]) == 0:
        raise Exception('motifs args is empty.')

    motif_len = len(motifs[0])
    count = {
        'A': [1]*motif_len,
        'C': [1]*motif_len,
        'G': [1]*motif_len,
        'T': [1]*motif_len
    }
    no_motifs = len(motifs)

    for i in range(no_motifs):
        for j in range(motif_len):
            symbol = motifs[i][j]
            count[symbol][j] += 1

    return count


def pseudo_profiles(motifs):
    '''
    Determine the profiles of the motifs by normalizing nucleotides frequencies,
    using Laplace's Succession
    '''

    if len(motifs) == 0 or len(motifs[0]) == 0:
        raise Exception('motifs args is empty.')
    
    motif_len = len(motifs[0])
    profiles = pseudo_count_motifs(motifs)
    sums = []

    for i in range(motif_len):
        sums.append(profiles['A'][i]+profiles['C'][i]+profiles['G'][i]+profiles['T'][i])

    sums = np.array(sums)

    for n in profiles:
        temp = np.array(profiles[n])
        profiles[n] = list(temp/(sums))

    return profiles


def pseudo_greedy_motif_search(dna, k, t):
    '''Greedily determine the best rated motifs for a dna sequence.'''

    best_motifs = [d[0:k] for d in dna]
    dna_len = len(dna[0])

    for i in range(dna_len-k+1):
        motifs = [dna[0][i:i+k]]

        for j in range(1, t):
            profile = pseudo_profiles(motifs[0:j])
            motifs.append(profile_most_probable_kmer(dna[j], k, profile))

        if score_consensus_motif(motifs) < score_consensus_motif(best_motifs):
            best_motifs = motifs

    return best_motifs


def motifs(profile, k, dna):
    '''Produce a list of profile-most probable k-mers from profile and dna sequences.'''

    k_mers = []

    for seq in dna:
        k_mers.append(profile_most_probable_kmer(seq, k, profile))

    return k_mers


def random_motifs(dna, k):
    '''Choose a random k-mer from each string in dna.'''

    motifs = []
    for seq in dna:
        seq_len = len(seq)
        idx = randint(0, seq_len-k)
        motifs.append(seq[idx:idx+k])

    return motifs


def randomized_motif_search(dna, k, t):
    '''
    Randomly generate a list of motifs until the scores associated with those
    motifs stops improving.
    '''

    temp_motifs = random_motifs(dna, k)
    best_motifs = temp_motifs

    while True:
        profile = pseudo_profiles(temp_motifs)
        temp_motifs = motifs(profile, k, dna)
        if score_consensus_motif(temp_motifs) < score_consensus_motif(best_motifs):
            best_motifs = temp_motifs
        else:
            return best_motifs
        

def normalize(probs: dict) -> dict:
    '''
    Normalize probabilties, so that their sum is equal to 1.0.
    
    Parameters
    ----------

    probs: dict
        A dictionary of probabilities associated with a k-mer.
        E.g. {'A': 0.1, 'C': 0.1, 'G': 0.1, 'T': 0.1}

    Returns
    -------
        A dictionary of normalized probabilities.
        E.g. {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
    '''

    prob_total = sum(list(probs.values()))

    norm_probs = probs.copy()
    for key in probs:
        norm_probs[key] = probs[key]/(prob_total)

    return norm_probs
    

def weighted_die(probs: dict) -> str:
    '''Return a randomly chosen weighted str from probs.'''

    options = list(probs.keys())
    weights = list(probs.values())
    k_mer = np.random.choice(options, p=weights)

    return str(k_mer)


def profile_generated_str(seq, profile, k):
    n = len(seq)
    probabilities = {}
    
    for i in range(0,n-k+1):
        probabilities[seq[i:i+k]] = profile_prob(seq[i:i+k], profile)

    probabilities = normalize(probabilities)
    
    return weighted_die(probabilities)


def gibbs_sampler(dna, k, t, n):
    '''
    Using the Gibbs Sampler algorithm, compute a set of motifs with high scoring values.
    '''
    t = len(dna)
    dna_len = len(dna[0])
    rand_k_mer_idx = np.random.choice(range(1, dna_len-k+1), size=t)
    motifs = []
    for seq, idx in zip(dna, rand_k_mer_idx):
        motifs.append(seq[idx:idx+k])

    best_motifs = motifs

    for j in range(1, n):
        i = np.random.choice(range(1,t), 1)[0]
        temp_motifs = []
        for idx, m in enumerate(motifs):
            if idx != i:
                temp_motifs.append(m)
        prof = pseudo_profiles(temp_motifs)
        motif_i = profile_generated_str(dna[i], prof, k)
        motifs[i] = motif_i

        if score_consensus_motif(motifs) < score_consensus_motif(best_motifs):
            best_motifs = motifs

    return best_motifs
