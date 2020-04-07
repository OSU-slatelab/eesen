. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. path.sh
. parse_options.sh

stage=3

#acoustic model parameters
am_nlayer=3
am_ncell_dim=320
am_model=resnet
am_window=1
am_subsample=1
#am_nproj=340
am_nproj=60
am_nproj_init=0
am_nfinalproj=100
am_norm=true
dropout=0.3
concatenate=5

dir_am=exp/train_char_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_concat${concatenate}

#lm_embed_size=64
#lm_batch_size=32
#lm_nlayer=1
#lm_ncell_dim=1024
#lm_drop_out=0.5
#lm_optimizer="adam"

#dir_lm=exp/train_lm_char_l${lm_nlayer}_c${lm_ncell_dim}_e${lm_embed_size}_d${lm_drop_out}_o${lm_optimizer}/

data=/data/corpora/CSLU_kids

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "                       Data Preparation                            "
  echo =====================================================================

  local/cslu_data_prep.sh $data  || exit 1;

  # Represent word spellings using a dictionary-like format
  local/cslu_prepare_phn_dict.sh || exit 1;
  local/cslu_prepare_char_dict.sh || exit 1;
fi


if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================

  fbankdir=fbank

  for set in train dev test; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit 1;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                Training AM with the Full Set                      "
  echo =====================================================================

  mkdir -p $dir_am

  echo generating train labels...
  python ./local/cslu_prepare_char_dict_tf.py --text_file ./data/train/text \
      --output_units ./data/local/dict_char/units.txt \
      --output_labels $dir_am/labels.tr || exit 1

  echo generating cv labels...
  python ./local/cslu_prepare_char_dict_tf.py --text_file ./data/dev/text \
      --output_units ./data/local/dict_char/units.txt \
      --output_labels $dir_am/labels.cv || exit 1

  python ./local/cslu_transition_probs.py --text_file $dir_am/labels.tr \
      --units ./data/local/dict_char/units.txt --out_file $dir_am/trans_prob.txt || exit 1

  # Train the network with CTC. Refer to the script for details about the arguments
  opts="--store_model --lstm_type=cudnn --batch_size=16"
  steps/train_ctc_tf.sh --train_opts "$opts" --lrscheduler halvsies --half_after 12 \
      --lr_rate 0.02 --nepoch 20 --l2 0.001 --target ctc --batch_norm $am_norm \
      --nhidden $am_ncell_dim --nproj $am_nproj --model $am_model --ninitproj $am_nproj_init \
      --nfinalproj $am_nfinalproj --nlayer $am_nlayer --dropout $dropout \
      --window $am_window --subsample $am_subsample --concatenate $concatenate \
      ./data/train ./data/dev $dir_am
fi
exit 0

if [ $stage -le 4 ]; then

  echo =====================================================================
  echo "                   Decoding using AM                      "
  echo =====================================================================

  epoch=final.ckpt
  filename=$(basename "$epoch")
  name_exp="${filename%.*}"
  #name_exp=./exp/train_char_l4_c320_mdeepbilstm_w3_nfalse_thomas/

  data=./data/test
  weights=$dir_am/model/$epoch
  config=$dir_am/model/config.pkl
  results=$dir_am/results/$name_exp
  labels=$dir_am/labels.ev

  python ./local/cslu_prepare_char_dict_tf.py --text_file ./data/test/text \
      --output_units ./data/local/dict_char/units.txt --output_labels $labels || exit 1

  ./steps/decode_ctc_am_tf.sh --config $config --data $data --weights $weights \
      --results $results --compute_ter true --labels $labels --target ctc
fi
