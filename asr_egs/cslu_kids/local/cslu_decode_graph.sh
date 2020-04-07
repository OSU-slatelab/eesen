#!/bin/bash

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph.

. ./path.sh || exit 1;

langdir=$1
lexicon=$2

order=3
lm_suffix="tg"  # We only use the trigram LMs. You can compile the 4gram ones similarly 
srilm_opts="-subset -prune-lowprobs -order $order"

# The CSLU LM
LM=data/local/lm/cslu.o${order}g.gz
outlangdir=${langdir}_cslu_$lm_suffix
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" $langdir $LM $lexicon $outlangdir

# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
# minimized.
fsttablecompose ${langdir}/L.fst $outlangdir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $outlangdir/LG.fst || exit 1;
fsttablecompose ${langdir}/T.fst $outlangdir/LG.fst > $outlangdir/TLG.fst || exit 1;
rm -rf $outlangdir/LG.fst

echo "Composing decoding graph TLG.fst succeeded"
