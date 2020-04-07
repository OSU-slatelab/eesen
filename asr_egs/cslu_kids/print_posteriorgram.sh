dir_am=exp/train_phn_l4_c512_bigru_ctc
epoch=final.ckpt
#utt=ksc0h250
utt=ks1187w0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh
. ./utils/parse_options.sh

filename=$(basename "$epoch")
name_exp="${filename%.*}"

data=./data/test
units=./data/local/dict_phn/units.txt
weights=$dir_am/model/$epoch
config=$dir_am/model/config.pkl

./steps/generate_posteriorgram.sh --config_file $config --data $data --units $units \
    --weights $weights --filename $dir_am/posteriorgram_${utt}.pdf --utterance $utt \
    --cutoff 0.1
