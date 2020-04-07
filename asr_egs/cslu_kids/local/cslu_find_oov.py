import argparse
import string

parser = argparse.ArgumentParser(description='Find out-of-vocabulary words for g2p')
parser.add_argument('--lexicon', default='data/local/dict_phn/lexicon.txt', type=str)
parser.add_argument('--text', default='data/train/text', type=str)
parser.add_argument('--output', default='oovs.txt', type=str)
args = parser.parse_args()

# Find words we already have a pronunciation for
lexicon_words = set()
for line in open(args.lexicon):
    lexicon_words.add(line.split()[0].lower())

# Load words that previous datasets have already found to be OOV
oov = set()
try:
    for line in open(args.output):
        oov.add(line.strip())
except:
    pass

# Add all OOV words in this text
for line in open(args.text):
    
    # Don't include utterance id
    line = line.split()[1:]

    for word in line:
        word = word.lower().strip(string.punctuation).strip()
        if word and word not in lexicon_words:
            oov.add(word)

# Write OOV words to file
with open(args.output, 'w', encoding="ascii") as w:
    for word in oov:
        w.write(word + '\n')
