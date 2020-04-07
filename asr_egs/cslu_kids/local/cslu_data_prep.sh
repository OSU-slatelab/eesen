#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

if [ $# -ne 1 ]; then
   echo "One argument: the cslu kids data directory."
   exit 1;
fi


dir=`pwd`/data/local/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for EESEN_ROOT
export PATH=$PATH:$EESEN_ROOT/tools/irstlm/bin
sph2pipe=$EESEN_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

cd $dir

# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.
rm -r links/ 2>/dev/null
mkdir links/
ln -s $* links

# Do some basic checks that we have what we expected.
if [ ! -d "links/CSLU_kids/labels" -o ! -d "links/CSLU_kids/speech" ]; then
  echo "cslu_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to CSLU_kids directory."
  exit 1;
fi

# Generate flist by making paths absolute and removing trailing number
#cat links/CSLU_kids/docs/{00,0[3-9],10}-verified\.txt | sed "s \.\. $* " | sed "s| [0-4]$||" | sort > train.flist
#cat links/CSLU_kids/docs/02-verified\.txt | sed "s \.\. $* " | sed "s| [0-4]$||" | sort > dev.flist
#cat links/CSLU_kids/docs/01-verified\.txt | sed "s \.\. $* " | sed "s| [0-4]$||" | sort > test.flist

cat links/CSLU_kids/docs/{00,0[3-9],10}-verified\.txt | sort > train.pre
cat links/CSLU_kids/docs/02-verified\.txt | sort > dev.pre
cat links/CSLU_kids/docs/01-verified\.txt | sort > test.pre

cat links/CSLU_kids/docs/{00,0[3-9],10}-verified\.txt | grep ' 1' | sort > train-clean.pre
cat links/CSLU_kids/docs/02-verified\.txt | sort > dev-clean.pre
cat links/CSLU_kids/docs/01-verified\.txt | sort > test-clean.pre

noiseword="<NOISE>"
for x in train train dev test; do
    for add in "" "-clean"; do
        x=$x$add
        cat $x.pre | sed "s \.\. $* " | sed "s| [0-4]$||" > $x.flist
        cat $x.pre | sed "s|.*/||" | sed "s|.wav||" > $x.verify
        $local/flist2scp.pl $x.flist | sort > ${x}_wav.scp
        cat ${x}_wav.scp | awk '{print $1}' | $local/find_transcripts.py links/CSLU_kids/docs/all.map > $x.trans1
        cat $x.trans1 | $local/normalize_transcript.pl $noiseword | sort > $x.txt || exit 1;
        cat ${x}_wav.scp | awk '{print $1}' | perl -ane 'chop; m:^.....:; print "$_ $&\n";' | sort > $x.utt2spk
        cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
    done
done

# Finally copy the data directories to data/
# Go back to the root directory
cd -
for x in train train dev test; do
    for add in "" "-clean"; do
        x=$x$add
        mkdir -p data/$x
        cp $dir/${x}_wav.scp data/$x/wav.scp || exit 1;
        cp $dir/$x.txt data/$x/text || exit 1;
        cp $dir/$x.spk2utt data/$x/spk2utt || exit 1;
        cp $dir/$x.utt2spk data/$x/utt2spk || exit 1;
        cp $dir/$x.verify data/$x/verify || exit 1;
    done
done

echo "Data preparation succeeded"
