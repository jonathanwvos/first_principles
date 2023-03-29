from utils import profile_prob
import numpy as np

NUCLEOTIDES = ['A', 'C', 'G', 'T']


if __name__ == '__main__':
    # profile = [
    #     [0.4,  0.3,  0.0,  0.1,  0.0,  0.9],
    #     [0.2,  0.3,  0.0,  0.4,  0.0,  0.1],
    #     [0.1,  0.3,  1.0,  0.1,  0.5,  0.0],
    #     [0.3,  0.1,  0.0,  0.4,  0.5,  0.0]
    # ]

    # mp_arr = np.array(profile).T
    # con_motif = ''

    # for m in mp_arr:
    #     max_profile = np.max(m)
    #     max_indices = np.where(m == max_profile)
    #     max_index = np.random.choice(max_indices[0], 1)
    #     con_motif += NUCLEOTIDES[max_index[0]]

    # print(con_motif)

    profile = {
        'A':  [0.4,  0.3,  0.0,  0.1,  0.0,  0.9],
        'C':  [0.2,  0.3,  0.0,  0.4,  0.0,  0.1],
        'G':  [0.1,  0.3,  1.0,  0.1,  0.5,  0.0],
        'T':  [0.3,  0.1,  0.0,  0.4,  0.5,  0.0]
    }

    print(profile_prob('GAGCTA', profile))