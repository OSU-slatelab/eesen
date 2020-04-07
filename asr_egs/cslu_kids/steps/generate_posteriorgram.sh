#!/bin/bash

batch_size=1
config_file=""
weights=""
data=""
use_priors=false
temperature=1
filename=posteriorgram.pdf
utterance=ks1308q0
units=""
cutoff=0.1

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

norm_vars=true
#tmpdir=`mktemp -d `
tmpdir=/scratch/tmp/plantingap/cslu

feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- |"

#copy-feats "${feats}" ark,scp:$tmpdir/f.ark,$tmpdir/test_local.scp

if $use_priors; then
    use_priors="--use_priors"
else
    use_priors=""
fi

$TF_PY -m generate_posteriorgram --data_dir $tmpdir --train_config $config_file --units $units \
    --trained_weights $weights --batch_size $batch_size --temperature $temperature \
    --filename $filename --utterance $utterance $use_priors --cutoff $cutoff

