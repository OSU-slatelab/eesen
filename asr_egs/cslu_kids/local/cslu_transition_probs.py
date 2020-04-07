#!/usr/bin/python

import argparse
import numpy as np

def main_parser():
    parser = argparse.ArgumentParser(description='Compute transition probs given input')
    parser.add_argument('--text_file', help = "input text file that will be translated to labels")
    parser.add_argument('--units', help = "list of units")
    parser.add_argument('--out_file', help = "input text file that will be translated to labels")
    return parser.parse_args()

def main():
    p = main_parser()

    units = []
    with open(p.units) as f:
        for line in f:
            units.append(line.split()[-1])


    trans_mat = np.zeros((len(units), len(units)+1), dtype = np.int32)

    with open(p.text_file) as f:
        for line in f:
            chars = [int(c) - 1 for c in line.split()[1:]]
            for i in range(len(chars)-1):
                trans_mat[chars[i], chars[i+1]] += 1
            trans_mat[chars[-1], len(units)] += 1

    with open(p.out_file, 'w') as f:
        for line in trans_mat:
            for i in range(len(line)-1):
                f.write("{} ".format(line[i]))
            f.write("{}\n".format(line[-1]))

if __name__ == '__main__':
    main()
