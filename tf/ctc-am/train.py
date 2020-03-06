#!/usr/bin/env python

"""
this project has been written following this naming convention:

https://google.github.io/styleguide/pyguide.html#naming
plus mutable vars in function (that are actually changes m_*)

"""

# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------

import argparse
import os
import constants
import os.path
import pickle
import sys
from eesen import Eesen
from utils.checkers import set_checkers
from utils.fileutils import debug

#from reader.sat_reader import sat_reader_factory
from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory


# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def main_parser():
    parser = argparse.ArgumentParser(description='Train TF-Eesen Model')

    #general arguments
    parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='enable debug mode')
    parser.add_argument('--store_model', default=False, dest='store_model', action='store_true', help='store model')
    parser.add_argument('--data_dir', default = "", help = "data dir")
    parser.add_argument('--train_dir', default = "", help = 'log and model (output) dir')
    parser.add_argument('--teacher_config', default = "", help = 'pretrained teacher model configuration file')
    parser.add_argument('--teacher_weights', default = "", help = 'pretrained teacher model weights dir')
    parser.add_argument('--dump_cv_fwd', default=False, dest='dump_cv_fwd', action='store_true', help='save forward pass of cv')
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--noshuffle', default=True, dest='do_shuf', action='store_false', help='do not shuffle training samples')
    parser.add_argument('--target', default='ctc', help='Options: "ctc", "detect", and "verify" (single label)')

    #ckpt arguments
    parser.add_argument('--import_config', default = "", help='load an old configuration file (config.pkl) extra labels will be added to old configuration')
    parser.add_argument('--continue_ckpt', default = "", help='continue this experiment')
    parser.add_argument('--diff_num_target_ckpt', default=False, action='store_true', help='change the number of targets after retaking training')
    parser.add_argument('--force_lr_epoch_ckpt', default=False, action='store_true', help='force to start for epoch 0 with the learning rate specified in flags')

    #augment arguments
    #parser.add_argument('--augment', default=False, dest='augment', action='store_true', help='do internal data augmentation')
    parser.add_argument('--window', default=3, type=int, help='how many frames will concatenate')
    parser.add_argument('--subsampling', default=3, type=int, help='how much subsampling will you apply')
    parser.add_argument('--roll', default=False, action='store_true', help='apply random rolls to the frames in the batch')
    parser.add_argument('--concatenate', default = 1, type = int, help = "How many utterances to concatenate")

    #architecture arguments
    parser.add_argument('--lstm_type', default="cudnn", help = "lstm type: cudnn, fuse, native")

    #TODO this should be done through a model manager
    parser.add_argument('--model', default="deepbilstm", help = "model: achen, bilstm, achen_sum")
    parser.add_argument('--nproj', default = 0, type=int, help='dimension of projection units, set to 0 if no projection needed')
    parser.add_argument('--ninitproj', default = 0, type=int, help='dimension of the initial projection layer, if 0 no initial projection layer will be added')
    parser.add_argument('--nfinalproj', default = 0, type=int, help='dimension of the final projection layer, if 0 no final projection layer will be added')

    parser.add_argument('--l2', default = 0.0, type=float, help='l2 normalization')
    parser.add_argument('--nlayer', default = 5, type=int, help='#layer')
    parser.add_argument('--nhidden', default = 320, type=int, help='dimension of hidden units in single direction')
    parser.add_argument('--clip', default = 5, type=float, help='gradient clipping')
    parser.add_argument('--batch_norm', default = False, dest='batch_norm', action='store_true', help='add batch normalization to FC layers')
    parser.add_argument('--grad_opt', default = "grad", help='optimizer: grad, adam, momentum, cuddnn only work with grad')
    parser.add_argument('--loss', default = "ctc", help='loss can be "ctc" or "eemmi"')

    #training runtime arguments
    parser.add_argument('--nepoch', default = 30, type=int, help='#epoch')
    parser.add_argument('--lrscheduler', default="halvsies", help = "lrscheduler: halvsies, newbob, constantlr") 
    parser.add_argument('--lr_spec', default = "", help='option specifier string to lrscheduler, overrides other command line options')
    parser.add_argument('--lr_rate', default = 0.03, type=float, help='learning rate')
    parser.add_argument('--min_lr_rate', default = 0.0005, type=float, help='minimal learning rate')
    parser.add_argument('--half_period', default = 4, type=int, help='half period in epoch of learning rate')
    parser.add_argument('--half_rate', default = 0.5, type=float, help='halving factor')
    parser.add_argument('--half_after', default = 8, type=int, help='halving becomes enabled after this many epochs')
    parser.add_argument('--dropout', default = 0.0, type=float, help='dropout, when set to 0, dropout is disabled, when set to 0.1 10% of the frames will be dropped out')
    parser.add_argument('--clip_norm', default = False, action='store_true', help='clip the gradient by norm')
    parser.add_argument('--kl_weight', default = 0.0, type=float, help='weight for label smoothing using kl divergence')

    #sat arguments
    parser.add_argument('--sat_type', default = constants.SAT_TYPE.UNADAPTED, help='apply and train a sat layer')
    parser.add_argument('--sat_stage', default = constants.SAT_SATGES.FINE_TUNE, help='apply and train a sat layer')
    parser.add_argument('--sat_nlayer', default = 2, type=int, help='number of sat layers for sat module')
    parser.add_argument('--continue_ckpt_sat', default = False, action='store_true', help='number of sat layers for sat module')

    return parser

def create_sat_config(args, config_imported = None):

    sat={}

    if config_imported and config_imported[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] == constants.SAT_TYPE.UNADAPTED :

        if config_imported[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] == constants.SAT_SATGES.UNADAPTED:
            sat[constants.CONF_TAGS.SAT_SATGE] = constants.SAT_SATGES.TRAIN_SAT

        elif config_imported[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] == constants.SAT_SATGES.TRAIN_SAT:
            sat[constants.CONF_TAGS.SAT_SATGE] = constants.SAT_SATGES.FINE_TUNE

        else:
            print("this sat stage ("+str(config_imported[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE])+") was not contemplates")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()
    else:
        #setting sat stage
        if args.sat_type == constants.SAT_TYPE.UNADAPTED:
            sat[constants.CONF_TAGS.SAT_TYPE] = constants.SAT_TYPE.UNADAPTED

        elif args.sat_type == constants.SAT_TYPE.CONCAT:

            sat[constants.CONF_TAGS.SAT_TYPE] = constants.SAT_TYPE.CONCAT

        elif args.sat_type == constants.SAT_TYPE.SHIFT:

            sat[constants.CONF_TAGS.SAT_TYPE] = constants.SAT_TYPE.SHIFT

        else:
            print("this sat type  ("+str(args.sat_type)+") was not expected")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

        if args.sat_stage == constants.SAT_SATGES.FINE_TUNE:
            sat[constants.CONF_TAGS.SAT_SATGE] = constants.SAT_SATGES.FINE_TUNE

        elif args.sat_stage == constants.SAT_SATGES.TRAIN_SAT:
            sat[constants.CONF_TAGS.SAT_SATGE] = constants.SAT_SATGES.TRAIN_SAT

        elif args.sat_stage == constants.SAT_SATGES.TRAIN_DIRECT:

            sat[constants.CONF_TAGS.SAT_SATGE] = constants.SAT_SATGES.TRAIN_DIRECT

        else:
            print("this sat type  ("+str(args.sat_stage)+") was not expected")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

    sat[constants.CONF_TAGS.NUM_SAT_LAYERS] = int(args.sat_nlayer)
    sat[constants.CONF_TAGS.CONTINUE_CKPT_SAT] = args.continue_ckpt_sat

    return sat

def create_online_arg_config(args):

    #TODO enter the values using a conf file or something
    online_augment_config={}

    if args.window % 2 == 0:
        raise ValueError("Window cannot be even")
    if args.roll:
        print("WARNING: --roll has been deprecated, option ignored")

    online_augment_config[constants.AUGMENTATION.WINDOW] = args.window
    online_augment_config[constants.AUGMENTATION.SUBSAMPLING] = args.subsampling
    #online_augment_config[constants.AUGMENTATION.ROLL] = args.roll
    online_augment_config[constants.AUGMENTATION.ROLL] = False
    online_augment_config[constants.AUGMENTATION.CONCATENATE] = args.concatenate

    return online_augment_config

def create_global_config(args):

    config = {
        #general arguments
        constants.CONF_TAGS.CONTINUE_CKPT: args.continue_ckpt,
        constants.CONF_TAGS.DIFF_NUM_TARGET_CKPT: args.diff_num_target_ckpt,
        constants.CONF_TAGS.FORCE_LR_EPOCH_CKPT: args.force_lr_epoch_ckpt,
        constants.CONF_TAGS.RANDOM_SEED: 15213,

        constants.CONF_TAGS.DEBUG: args.debug,
        constants.CONF_TAGS.STORE_MODEL: args.store_model,
        constants.CONF_TAGS.DATA_DIR: args.data_dir,
        constants.CONF_TAGS.TRAIN_DIR: args.train_dir,
        constants.CONF_TAGS.TEACHER_CONFIG: args.teacher_config,
        constants.CONF_TAGS.TEACHER_WEIGHTS: args.teacher_weights,
        constants.CONF_TAGS.DUMP_CV_FWD: args.dump_cv_fwd,
        constants.CONF_TAGS.TARGET: args.target,

        #io arguments
        constants.CONF_TAGS.BATCH_SIZE: args.batch_size,
        constants.CONF_TAGS.DO_SHUF: args.do_shuf,

        #training runtime arguments
        constants.CONF_TAGS.NEPOCH: args.nepoch,
        constants.CONF_TAGS.LRSCHEDULER: args.lrscheduler,
        constants.CONF_TAGS.LR_SPEC: args.lr_spec,
        constants.CONF_TAGS.LR_RATE: args.lr_rate,
        constants.CONF_TAGS.MIN_LR_RATE: args.min_lr_rate,
        constants.CONF_TAGS.HALF_PERIOD: args.half_period,
        constants.CONF_TAGS.HALF_RATE: args.half_rate,
        constants.CONF_TAGS.HALF_AFTER: args.half_after,
        constants.CONF_TAGS.DROPOUT: args.dropout,
        constants.CONF_TAGS.CLIP_NORM: args.clip_norm,
        constants.CONF_TAGS.KL_WEIGHT: args.kl_weight,
        constants.CONF_TAGS.COMPUTE_ACC: False,

        #architecture arguments
        #TODO this can be joined with one argument
        constants.CONF_TAGS.MODEL: args.model,
        constants.CONF_TAGS.LSTM_TYPE: args.lstm_type,
        constants.CONF_TAGS.NPROJ: args.nproj,
        constants.CONF_TAGS.FINAL_NPROJ: args.nfinalproj,
        constants.CONF_TAGS.INIT_NPROJ: args.ninitproj,
        constants.CONF_TAGS.L2: args.l2,
        constants.CONF_TAGS.NLAYERS: args.nlayer,
        constants.CONF_TAGS.NHIDDEN: args.nhidden,
        constants.CONF_TAGS.CLIP: args.clip,
        constants.CONF_TAGS.BATCH_NORM: args.batch_norm,
        constants.CONF_TAGS.GRAD_OPT: args.grad_opt,
        constants.CONF_TAGS.LOSS: args.loss,

    }

    config[constants.CONF_TAGS.SAT_CONF] = create_sat_config(args)
    config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF] = create_online_arg_config(args)

    return config

def update_conf_import(config, args):

    if args.data_dir:
        config[constants.CONF_TAGS.DATA_DIR] = args.data_dir

    if args.train_dir:
        config[constants.CONF_TAGS.TRAIN_DIR] = args.train_dir

    if args.teacher_weights:
        config[constants.CONF_TAGS.TEACHER_WEIGHTS] = args.teacher_weights

    if args.teacher_config:
        config[constants.CONF_TAGS.TEACHER_CONFIG] = args.teacher_config

    if args.dump_cv_fwd:
        config[constants.CONF_TAGS.DUMP_CV_FWD] = args.dump_cv_fwd

    if args.continue_ckpt:
        config[constants.CONF_TAGS.CONTINUE_CKPT] = args.continue_ckpt

    config[constants.CONF_TAGS.LRSCHEDULER] = args.lrscheduler
    config[constants.CONF_TAGS.NEPOCH] = args.nepoch
    config[constants.CONF_TAGS.LR_RATE] = args.lr_rate
    config[constants.CONF_TAGS.MIN_LR_RATE] = args.min_lr_rate

def import_config(args):

    if not os.path.exists(args.import_config):
        print("Error: path_config does not correspond to a valid path: "+args.import_config)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    config = pickle.load(open(args.import_config, "rb"))
    update_conf_import(config, args)
    config[constants.CONF_TAGS.FORCE_LR_EPOCH_CKPT] = args.force_lr_epoch_ckpt
    config[constants.CONF_TAGS.DIFF_NUM_TARGET_CKPT] = args.diff_num_target_ckpt
    config[constants.CONF_TAGS.SAT_CONF] = create_sat_config(args, config)

    return config

# -----------------------------------------------------------------
#   Main part
# -----------------------------------------------------------------

def main():
    #TODO construct a factory/helper to load everything by just looking at data_dir

    parser = main_parser()
    args = parser.parse_args()

    if args.import_config:
        config = create_global_config(args)
        config.update(import_config(args))
    else:
        config = create_global_config(args)

    tr_data = {}
    cv_data = {}

    # load feats
    data_format = 'kaldi'
    if config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ARCNET_VIDEO:
        data_format = 'video'

    tr_data['x'] = feats_reader_factory.create_reader('train', data_format, config)
    cv_data['x'] = feats_reader_factory.create_reader('cv', data_format, config)
    tr_ids = tr_data['x'].get_batches_id()
    cv_ids = cv_data['x'].get_batches_id()

    # load targets
    tr_args = ['labels.tr', 'txt', config, tr_ids]
    cv_args = ['labels.cv', 'txt', config, cv_ids]
    if args.import_config:
        tr_args.append(config[constants.CONF_TAGS.LANGUAGE_SCHEME])
        cv_args.append(config[constants.CONF_TAGS.LANGUAGE_SCHEME])

    tr_data['y'] = labels_reader_factory.create_reader(*tr_args)
    cv_data['y'] = labels_reader_factory.create_reader(*cv_args)

    # Load verify
    if args.target == 'verify':
        tr_data['z'] = labels_reader_factory.create_reader('verify.tr', 'txt', config, tr_ids)
        cv_data['z'] = labels_reader_factory.create_reader('verify.cv', 'txt', config, cv_ids)

    #set config (targets could change)
    config[constants.CONF_TAGS.INPUT_FEATS_DIM] = cv_data['x'].get_num_dim()
    config[constants.CONF_TAGS.LANGUAGE_SCHEME] = cv_data['y'].get_language_scheme()

    if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] != constants.SAT_TYPE.UNADAPTED:

        tr_data['sat'] = sat_reader_factory.create_reader('kaldi', config, tr_ids)
        cv_data['sat'] = sat_reader_factory.create_reader('kaldi', config, cv_ids)


        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM] = int(tr_sat.get_num_dim())
        config[constants.CONF_TAGS.MODEL_DIR] = os.path.join(
            config[constants.CONF_TAGS.TRAIN_DIR],
            constants.DEFAULT_NAMES.MODEL_DIR_NAME,
            constants.DEFAULT_NAMES.SAT_DIR_NAME+"_"+
                config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE]+"_"+
                config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE]+"_"+
                str(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.NUM_SAT_LAYERS])
        )

        print("adaptation data with a dimensionality of {} prepared...\n".format(
            str(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM])))
    else:
        config[constants.CONF_TAGS.MODEL_DIR] = os.path.join(config[constants.CONF_TAGS.TRAIN_DIR],
                                                             constants.DEFAULT_NAMES.MODEL_DIR_NAME)

    data = (tr_data, cv_data)

    # checking that all sets are consistent
    set_checkers.check_sets_training(tr_data, cv_data)

    # create folder for storing experiment
    if not os.path.exists(config[constants.CONF_TAGS.MODEL_DIR]):
        os.makedirs(config[constants.CONF_TAGS.MODEL_DIR])

    pickle.dump(config, open(os.path.join(config[constants.CONF_TAGS.MODEL_DIR], "config.pkl"), "wb"), protocol=4)

    # start the actual training
    eesen = Eesen()

    print(80 * "-")
    print("done with data preparation")
    print(80 * "-")
    #print("begining training with following config:")

    #for key, value in config.items():
    #    print(key+" "+str(value))
    #print(80 * "-")

    eesen.train(data, config)


if __name__ == "__main__":
    main()

