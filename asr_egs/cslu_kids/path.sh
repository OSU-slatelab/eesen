export EESEN_ROOT=/u/plantinga.1/Documents/Repositories/eesen_tf_clean
export EESEN_EGS_ROOT=$EESEN_ROOT/egs
#export LD_LIBRARY_PATH=/u/drspeech/opt/cuda-8.0/lib64/:$LD_LIBRARY_PATH #/u/hey/lib/cuda-6.5/lib64/:$LD_LIBRARY_PATH   # osu. Use the shared /u/hey/lib instead of /usr/local so that every machine can be used when GPU isn't required.
export PATH=$PWD/utils/:$EESEN_ROOT/src/lmbin/:$EESEN_ROOT/src/bin/:$EESEN_ROOT/tools/sph2pipe_v2.5/:$EESEN_ROOT/src/bin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/extras/irstlm/bin/:$EESEN_ROOT/src/fstbin/:$EESEN_ROOT/src/gmmbin/:$EESEN_ROOT/src/featbin/:$EESEN_ROOT/src/lm/:$EESEN_ROOT/src/sgmmbin/:$EESEN_ROOT/src/sgmm2bin/:$EESEN_ROOT/src/fgmmbin/:$EESEN_ROOT/src/latbin/:$EESEN_ROOT/src/nnetbin:$EESEN_ROOT/src/nnet2bin/:$EESEN_ROOT/src/kwsbin:$PWD:$PATH
TMPDIR=/tmp
export LC_ALL=C
export IRSTLM=$EESEN_ROOT/tools/extras/irstlm

export PATH=${PATH}:${IRSTLM}/bin
export LIBLBFGS=/homes/3/bagchid/kaldi20160818/tools/liblbfgs-1.10
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIBLBFGS}/lib/.libs
export SRILM=/homes/3/bagchid/kaldi20160818/tools/srilm
export PATH=${PATH}:${SRILM}/bin:${SRILM}/bin/i686-m64

export PATH=$PATH:$EESEN_ROOT/src/decoderbin/:$EESEN_ROOT/src/netbin/

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64/atlas/
export LD_LIBRARY_PATH="/u/drspeech/opt/anaconda3/lib:${LD_LIBRARY_PATH}"

export PYTHONPATH=${PYTHONPATH}:/u/plantinga.1/Documents/Repositories/eesen_tf_clean/tf/ctc-am/
#export PYTHONHOME=/u/plantinga.1/.conda/envs/tf/
export TF_PY='srun -q -J eesen -w vibranium gpurun.sh -c 1 /u/plantinga.1/.conda/envs/tf-1.13/bin/python'
#export TF_PY='gpurun.sh -c 0 /u/plantinga.1/.conda/envs/tf/bin/python'
