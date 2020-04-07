#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
# Copyright 2016  Florian Metze (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models using tensorflow

## Begin configuration section

#main calls and arguments
#train_tool="srun -q --gres=gpu:1 -J eesen -w adamantium $TF_PY -m train"
train_tool="$TF_PY -u -m train"
train_opts="--store_model --lstm_type=cudnn"

#network architecture
model="deepbilstm"

nlayer=3
nhidden=320

nproj=""
nfinalproj=
ninitproj=""

#speaker adaptation configuration
sat_type=""
sat_stage=""
sat_path=""
sat_nlayer=2
continue_ckpt_sat=false

#training configuration
nepoch=""
lr_rate=""
lrscheduler=""
lr_spec=""
dropout=""
kl_weight=""
debug=false
half_after=""
target=""

#continue training
continue_ckpt=""
diff_num_target_ckpt=false
force_lr_epoch_ckpt=false

#training options
deduplicate=false
l2=0.0
batch_norm=true
grad_opt=grad
loss=ctc
teacher_config=""
teacher_weights=""

# augmentation
window=3
subsample=3
concatenate=1

if $deduplicate; then
    deduplicate="--deduplicate"
else
    deduplicate=""
fi

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

echo $train_opts

#checking number of arguments
if [ $# != 3 ]; then
   echo $1
   echo "Usage: $0 <data-tr> <data-cv> <exp-dir>"
   echo " e.g.: $0 data/train_tr data/train_cv exp/train_phn"
   exit 1;
fi

#getting main arguments
data_tr=$1
data_cv=$2
dir=$3

#creating tmp directory (concrete tmp path is defined in path.sh)
#tmpdir=`mktemp -d`
tmpdir=/scratch/tmp/plantingap/cslu

#trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir &" ERR EXIT

#checking folders
for f in $data_tr/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo `basename "$0"`": no such file $f" && exit 1;
done


## Adjust parameter variables

if $debug; then
    debug="--debug"
else
    debug=""
fi

if $force_lr_epoch_ckpt; then
    force_lr_epoch_ckpt="--force_lr_epoch_ckpt"
else
    force_lr_epoch_ckpt=""
fi

if $diff_num_target_ckpt; then
    diff_num_target_ckpt="--diff_num_target_ckpt"
else
    diff_num_target_ckpt=""
fi

if [[ $continue_ckpt != "" ]]; then
    continue_ckpt="--continue_ckpt $continue_ckpt"
else
    continue_ckpt=""
fi

if [ -n "$ninitproj" ]; then
    ninitproj="--ninitproj $ninitproj"
fi

if [ -n "$nfinalproj" ]; then
    nfinalproj="--nfinalproj $nfinalproj"
fi

if [ -n "$nproj" ]; then
    nproj="--nproj $nproj"
fi

if [ -n "$nepoch" ]; then
    nepoch="--nepoch $nepoch"
fi

if [ -n "$dropout" ]; then
    dropout="--dropout $dropout"
fi

if [ -n "$kl_weight" ]; then
    kl_weight="--kl_weight $kl_weight"
fi

if [ -n "$lr_rate" ]; then
    lr_rate="--lr_rate $lr_rate"
fi

if [[ "$lrscheduler" != "" ]]; then
    lrscheduler="--lrscheduler $lrscheduler"
fi

if [[ "$lr_spec" != "" ]]; then
    lr_spec="--lr_spec $lr_spec"
fi

if [[ "$recognizer_dir" != "" ]]; then
    recognizer_dir="--recognizer_dir $recognizer_dir"
fi

#TODO solvME!
if [ -n "$half_after" ]; then
    half_after="--half_after $half_after"
fi

subsampling=`echo $train_opts | sed 's/.*--subsampling \([0-9]*\).*/\1/'`

if [[ "$subsampling" == [0-9]* ]]; then
    #this is needed for the filtering - let's hope this value is correct
    :
else
    subsampling=1
fi


if [[ "$roll" == "true" ]]; then
# --roll is deprecated
#    roll="--roll"
    echo "WARNING: --roll is deprecated, ignoring option"
    roll=""
fi

if [ -n "$l2" ]; then
    l2="--l2 $l2"
fi

if [[ "$batch_norm" == "true" ]]; then
    batch_norm="--batch_norm"
else
    batch_norm=""
fi

#SPEAKER ADAPTATION

if [[ "$sat_type" != "" ]]; then
    copy-feats ark:$sat_path ark,scp:$tmpdir/sat_local.ark,$tmpdir/sat_local.scp
    sat_type="--sat_type $sat_type"
else
    sat_type=""
fi

if [[ "$sat_stage" != "" ]]; then
    sat_stage="--sat_stage $sat_stage"
else
    sat_stage=""
fi

if $continue_ckpt_sat; then
    continue_ckpt_sat="--continue_ckpt_sat"
else
    continue_ckpt_sat=""
fi

sat_nlayer="--sat_nlayer $sat_nlayer"

if [[ "$dump_cv_fwd" == "true" ]]; then
    dump_cv_fwd="--dump_cv_fwd"
else
    dump_cv_fwd=""
fi

if [[ "$target" != "" ]]; then
    target="--target $target"
else
    target=""
fi

window="--window $window"
subsample="--subsampling $subsample"
concatenate="--concatenate $concatenate"

if [[ "$teacher_config" != "" ]]; then
    teacher_config="--teacher_config $teacher_config"
else
    teacher_config=""
fi

if [[ "$teacher_weights" != "" ]]; then
    teacher_weights="--teacher_weights $teacher_weights"
else
    teacher_weights=""
fi

echo ""
echo copying cv features ...
echo ""

data_tr=$1
data_cv=$2

#feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- |"
#copy-feats "$feats_cv" ark,scp:$tmpdir/cv.ark,$tmpdir/cv_tmp.scp || exit 1;

echo ""
echo copying training features ...
echo ""

#feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- |"
#copy-feats "$feats_tr" ark,scp:$tmpdir/train.ark,$tmpdir/train_tmp.scp || exit 1;

echo ""
echo copying labels ...
echo ""

if [ -f $dir/labels.tr.gz ] && [ -f $dir/labels.cv.gz ] ; then
    gzip -cd $dir/labels.tr.gz > $tmpdir/labels.tr || exit 1
    gzip -cd $dir/labels.cv.gz > $tmpdir/labels.cv || exit 2
elif [ -f $dir/labels.tr ] && [ -f $dir/labels.cv ] ; then
    cp $dir/labels.tr $tmpdir
    cp $dir/labels.cv $tmpdir
else
    echo error, labels not found...
    echo exiting...
    exit 1
fi

cp $dir/verify.tr $tmpdir
cp $dir/verify.cv $tmpdir
#cp $dir/trans_prob.txt $tmpdir

# Compute the occurrence counts of labels in the label sequences.
# These counts will be used to derive prior probabilities of the labels.
awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' $tmpdir/labels.tr | \
  analyze-counts --verbose=0 --binary=false ark:- $dir/label.counts > $dir/labelcounts.log || exit 1

echo ""
echo cleaning train set ...
echo ""

for f in $tmpdir/*.tr; do

	echo ""
	echo cleaning train set $(basename $f)...
	echo ""

	python ./utils/clean_length.py --scp_in $tmpdir/train_tmp.scp --labels $f \
	       --subsampling $subsampling --scp_out $tmpdir/train_local.scp $deduplicate
done

for f in $tmpdir/*.cv; do

    echo ""
    echo cleaning cv set $(basename $f)...
    echo ""

    python ./utils/clean_length.py --scp_in  $tmpdir/cv_tmp.scp --labels $f \
	   --subsampling $subsampling --scp_out $tmpdir/cv_local.scp 
done

#path were cache cuda binaries will be compiled and stored
#export CUDA_CACHE_PATH=$tmpdir
export CUDA_CACHE_PATH=/scratch/tmp/plantingap/cuda_cache


cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"

cmd="$train_tool $train_opts \
    --model $model --nlayer $nlayer --nhidden $nhidden $ninitproj $nproj $nfinalproj $nepoch \
    $dropout $lr_rate $lrscheduler $lr_spec $l2 $batch_norm --train_dir $dir --data_dir $tmpdir \
    $kl_weight $half_after $sat_stage $sat_type $sat_nlayer $debug $continue_ckpt $continue_ckpt_sat \
    $diff_num_target_ckpt $force_lr_epoch_ckpt $dump_cv_fwd $target $window $subsample $concatenate \
    $recognizer_dir --grad_opt $grad_opt --loss $loss $teacher_config $teacher_weights"

echo $cmd

$cmd || exit 1;

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING ENDS [$cur_time]"
