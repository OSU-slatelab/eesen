#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh

stage=4
data=/data/corpora/CSLU_kids

. utils/parse_options.sh

# add check for IRSTLM prune-lm
if ! prune-lm > /dev/null 2>&1; then
    echo "Error: prune-lm (part of IRSTLM) is not in path"
    echo "Make sure that you run tools/extras/install_irstlm.sh in the main Eesen directory;"
    echo " this is no longer installed by default."
    exit 1
fi

# Specify network structure and generate the network topology
#input_feat_dim=120 # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
#lstm_layer_num=4   # number of LSTM layers
#lstm_cell_dim=320  # number of memory cells in every LSTM layer

#acoustic model parameters
am_nlayer=4
am_nproj=100
am_ninitproj=60
am_nfinalproj=0
am_ncell_dim=512

am_model=gru
clean=true
loss=ctc_avg3_ts_bitsua
teacher_config="--teacher_config exp/train_phn_l4_c512_bigru_ctc_align_ts_uni/model/config.pkl"
teacher_weights="--teacher_weights exp/train_phn_l4_c512_bigru_ctc_align_ts_uni/model/final.ckpt"

concat=1

lr_spec="lr_rate=0.05,half_after=8,nepoch=25"

dir=exp/train_phn_l${am_nlayer}_c${am_ncell_dim}_${am_model}_${loss}

#[[ $clean == true ]] && dir=${dir}_clean

mkdir -p $dir

#recognizer_dir=/data/data4/scratch/plantingap/eesen_tf_clean_cslu_kids/exp/train_phn_l4_c320/model

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================
  # Use the same datap prepatation script from Kaldi
  local/cslu_data_prep.sh $data

  # Construct the phoneme-based lexicon from the CMU dict
  local/cslu_prepare_phn_dict.sh || exit 1;

  # Compile the lexicon and token FSTs
  utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  # Make LM stuffs
  echo generating train labels...
  python ./local/cslu_prepare_phn_dict.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train/text --output_units ./data/local/dict_phn/units.txt --output_labels $dir/labels.tr || exit 1
  python ./local/cslu_prepare_phn_dict.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train-clean/text --output_units ./data/local/dict_phn/units.txt --output_labels $dir/labels.cl || exit 1
  cp ./data/train/verify $dir/verify.tr
  [[ $clean == true ]] && cp ./data/train-clean/verify $dir/verify.tr

  echo creating graph...
  local/cslu_train_lms.sh data/train/text data/local/dict_phn/lexicon.txt data/local/lm
  local/cslu_decode_graph.sh data/lang_phn data/local/dict_phn/lexicon.txt

  echo generating dev labels...
  python ./local/cslu_prepare_phn_dict.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/dev/text --output_units ./data/local/dict_phn/units.txt --output_labels $dir/labels.cv || exit 1
  cp ./data/dev/verify $dir/verify.cv

  echo generating eval labels...
  python ./local/cslu_prepare_phn_dict.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/test/text --output_units ./data/local/dict_phn/units.txt --output_labels $dir/labels.ev || exit 1
  cp ./data/test/verify $dir/verify.ev
  
  python ./local/cslu_transition_probs.py --text_file $dir/labels.tr \
    --units ./data/local/dict_phn/units.txt --out_file $dir/trans_prob.txt || exit 1

fi

if [ $stage -le 0 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  #utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train data/train_tr95 data/train_cv05 || exit 1

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  #for set in train_tr95 train_cv05; do
  #  steps/make_fbank.sh --cmd "$train_cmd" --nj 14 data/$set exp/make_fbank/$set $fbankdir || exit 1;
  #  utils/fix_data_dir.sh data/$set || exit;
  #  steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  #done

  for set in train train-clean dev test; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                      Training CTC network               "
  echo =====================================================================

  train_data_dir=data/train

  [[ $clean == true ]] && train_data_dir=${train_data_dir}-clean

  cp exp/train_phn_l4_c512_bigru_ctc/label* $dir/
  cp exp/train_phn_l4_c512_bigru_ctc/verify* $dir/

  # Train the network with CTC. Refer to the script for details about the arguments
  ( 
    steps/train_ctc_tf.sh --l2 0.001 --batch_norm false --nlayer $am_nlayer --nhidden $am_ncell_dim \
      --lrscheduler halvsies --lr_spec $lr_spec --model $am_model --ninitproj $am_ninitproj --loss $loss \
      --nproj $am_nproj --nfinalproj $am_nfinalproj --concatenate $concat --grad_opt adam --dropout 0.2 \
      $teacher_config $teacher_weights \
      $train_data_dir data/dev $dir 2>&1 || exit 1
  ) | tee $dir/train.log 

fi

if [ $stage -le 4 ]; then

  echo =====================================================================
  echo "            Decoding using AM + WFST decoder                     "
  echo =====================================================================

  # globals used by stages 5

  epoch=final.ckpt
  filename=$(basename "$epoch")
  name_exp="${filename%.*}"

  data=./data/test
  weights=$dir/model/$epoch
  config=$dir/model/config.pkl
  results=$dir/results_${name_exp}
  units=./data/lang_phn/units.txt
  labels=$dir/labels.ev
  verify=$dir/verify.ev


  # Temporary variables for debugging
  #bs=5.0
  #lm_suffix=test_tg

  #for lm_suffix in test_tg test_tgpr; do
  #    for bs in 4.0 5.0 6.0 7.0; do
  #    ./steps/decode_ctc_lat_tf.sh \
  #        --model $weights \
  #        --nj 1 \
  #        --blank_scale $bs \
  #        ./data/lang_phn_${lm_suffix} \
  #        ${data} \
  #        ${results}_bs${bs}_${lm_suffix}
  #done

  #./steps/decode_ctc_lat_tf.sh \
  #    --model $weights \
  #    --nj 1 \
  #    --blank_scale 5.0 \
  #    --scoredir $dir/score \
  #    ./data/lang_phn_cslu_tg \
  #    $data \
  #    $results

  ./steps/decode_ctc_am_tf.sh --config_file $config --data $data --weights $weights \
      --results $results --compute_ter true --compute_acc true --labels $labels --verify_file $verify

  #./steps/generate_posteriorgram.sh --config $config --data $data --units $units \
  #    --weights $weights --filename phone_posteriorgram.pdf --utterance ks1308q0
fi


