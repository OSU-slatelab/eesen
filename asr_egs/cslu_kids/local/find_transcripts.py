#!/usr/bin/python

import sys, string

with open(sys.argv[1]) as f:
    transcript_map = {}
    for line in f:
        line = line.split()
        if len(line) > 1:
            transcript_map[line[0]] = " ".join([x.strip('"') for x in line[1:]])

for line in sys.stdin:
    utt_id = line[-4:-2].upper()
    print("%s %s" % (line.rstrip(), transcript_map[utt_id]))


