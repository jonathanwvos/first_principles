import re


def gc_content(seq):
    '''Determine the GC content in the DNA/RNA seq.'''
    seq = seq.upper()

    g = seq.count('G')
    c = seq.count('C')

    return (g+c)/len(seq)


if __name__ == '__main__':
    sequences = {}

    with open('rosalind_fasta.txt', 'r') as f:
        text = f.read()
        regex = r'>(Rosalind_\d{4})\n([ATGC]+)\n?'

        matches = re.findall(regex, text)
        gcs = {}

        for key, seq in matches:
            # gcs[key] = gc_content(seq)
            print(seq)
            # print(key, '\n', gc_content(seq)*100, '\n')

        
        # text = text.replace('\n', '')

        # items = text.split('>')

        # print(items)