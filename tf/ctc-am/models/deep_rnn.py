import constants
import tensorflow as tf
from utils.fileutils import debug
import sys
import numpy as np

class DeepRNN:

    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length

    def batch_norm_fn(self, inputs, scope = "bn"):
        return tf.contrib.layers.batch_norm(
            inputs,
            scope = scope,
            center = True,
            scale = True,
            decay = 0.9,
            is_training = self.is_training_ph,
            updates_collections = None,
        )

    def my_cudnn_lstm(self, outputs, scope = "cudnn_lstm"):
        """
        outputs: time, batch_size, feat_dim
        """
        with tf.variable_scope(scope):

            kwargs = {
                'num_layers' : 1 if self.nproj > 0 else self.nlayer,
                'num_units' : self.nhidden,
                'input_mode' : 'linear_input',
                'direction' : self.direction,
                'kernel_initializer' : tf.contrib.layers.xavier_initializer(),
                'bias_initializer' : tf.contrib.layers.xavier_initializer(),
                #'dropout' : self.dropout,
            }

            if self.nproj > 0:
                for i in range(self.nlayer):
                    with tf.variable_scope("layer%d" % i):

                        if self.kernel == 'LSTM':
                            cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(**kwargs)
                        elif self.kernel == 'GRU':
                            cudnn_model = tf.contrib.cudnn_rnn.CudnnGRU(**kwargs)

                        outputs, _ = cudnn_model(outputs, training = self.is_training)

                        outputs = tf.layers.dropout(outputs, rate=self.dropout, training=self.is_training_ph)
                        outputs = tf.contrib.layers.fully_connected(
                            activation_fn = None,
                            inputs = outputs,
                            num_outputs = self.nproj,
                            scope = "intermediate_projection",
                        )
                        outputs = tf.layers.dropout(outputs, rate=self.dropout, training=self.is_training_ph)

                        if self.batch_norm:
                            outputs = self.batch_norm_fn(outputs)
            else:

                if self.kernel == 'LSTM':
                    cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(**kwargs)
                elif self.kernel == 'GRU':
                    cudnn_model = tf.contrib.cudnn_rnn.CudnnGRU(**kwargs)

                outputs, _output_h = cudnn_model(outputs, None, True)

                if self.batch_norm:
                    outputs = self.batch_norm_fn(outputs)

        return outputs

    def my_fuse_block_lstm(self, outputs, scope):
        """
        outputs: time, batch_size, feat_dim
        """
        with tf.variable_scope(scope):
            for i in range(self.nlayer):
                with tf.variable_scope("layer%d" % i):
                    with tf.variable_scope("fw_lstm"):
                        fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(self.nhidden, cell_clip = 0)
                        fw_out, _ = fw_lstm(outputs, dtype=tf.float32, sequence_length = self.seq_len)
                    with tf.variable_scope("bw_lstm"):
                        bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(tf.contrib.rnn.LSTMBlockFusedCell(self.nhidden, cell_clip = 0))
                        bw_out, _ = bw_lstm(outputs, dtype=tf.float32, sequence_length = self.seq_len)
                    outputs = tf.concat_v2([fw_out, bw_out], 2, name = "output")
                    # outputs = tf.concat([fw_out, bw_out], 2, name = "output")
                    if self.nproj > 0:
                        outputs = tf.contrib.layers.fully_connected(
                            activation_fn = None, inputs = outputs,
                            num_outputs = self.nproj, scope = "projection")
        return outputs

    def my_native_lstm(self, outputs, scope):
        """
        outputs: time, batch_size, feat_dim
        """
        with tf.variable_scope(scope):
            for i in range(self.nlayer):
                with tf.variable_scope("layer%d" % i):
                    if self.nproj > 0:
                        cell = tf.contrib.rnn.LSTMCell(self.nhidden, num_proj = self.nproj, state_is_tuple = True)
                    else:
                        cell = tf.contrib.rnn.BasicLSTMCell(self.nhidden, state_is_tuple = True)

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs,
                        self.seq_len, time_major = True, dtype = tf.float32)

                    outputs = tf.concat_v2(values = outputs, axis = 2, name = "output")
        return outputs

    def my_sat_layers(self, num_sat_layers, adapt_dim, outputs):

        for i in range(num_sat_layers-1):
            with tf.variable_scope("layer%d" % i):
                outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = adapt_dim)

        with tf.variable_scope("last_sat_layer"):
            outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = self.nfeat)

        return outputs


    def my_sat_module(self, config, input_feats, input_sat):


        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                == constants.SAT_SATGES.TRAIN_SAT:

            self.is_trainable_sat=False

        with tf.variable_scope(constants.SCOPES.SPEAKER_ADAPTAION):

            if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] == constants.SAT_TYPE.CONCAT:

                with tf.variable_scope(constants.SCOPES.SAT_FUSE):
                    sat_input = tf.tile(input_sat, tf.stack([tf.shape(input_feats)[0], 1, 1]))
                    outputs = tf.concat([input_feats, sat_input], 2)

                    return self.my_sat_layers(
                        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.NUM_SAT_LAYERS],
                        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM],
                        config[constants.CONF_TAGS.INPUT_FEATS_DIM],
                        outputs,
                    )

            elif config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] == constants.SAT_TYPE.SHIFT:

                    learned_sat = self.my_sat_layers(
                        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.NUM_SAT_LAYERS],
                        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM],
                        config[constants.CONF_TAGS.INPUT_FEATS_DIM],
                        input_sat,
                    )

                    return tf.add(input_feats, learned_sat, name="shift")
            else:
                sat_type = config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE]
                raise ValueError("this sat type ("+str(sat_type)+") was not contemplates")

    def make_layer(self, inputs, nfilt, name):

        outputs = tf.layers.conv2d(
            inputs = inputs,
            filters = nfilt,
            kernel_size = 3,
            strides = (1, 2),
            activation = tf.nn.relu,
            padding = 'same',
            name = name,
        )

        outputs = tf.layers.dropout(
            inputs = outputs,
            rate = self.dropout,
            training = self.is_training_ph,
            noise_shape = [self.batch_size, 1, 1, nfilt],
        )

        return outputs


    def __init__(self, config, scope, teacher = None, train = True):

        self.nfeat = config[constants.CONF_TAGS.INPUT_FEATS_DIM]
        self.nhidden = config[constants.CONF_TAGS.NHIDDEN]
        self.nlayer = config[constants.CONF_TAGS.NLAYERS]
        self.nproj = config[constants.CONF_TAGS.NPROJ]
        self.dropout = config[constants.CONF_TAGS.DROPOUT]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        loss_fn = config[constants.CONF_TAGS.LOSS]

        bi_list = [
            constants.MODEL_NAME.DEEP_BILSTM,
            constants.MODEL_NAME.CNN_BILSTM,
            constants.MODEL_NAME.DEEP_BIGRU,
            constants.MODEL_NAME.CNN_BIGRU,
        ]

        conv_list = [
            constants.MODEL_NAME.CNN_BILSTM,
            constants.MODEL_NAME.CNN_BIGRU,
            constants.MODEL_NAME.CNN_LSTM,
            constants.MODEL_NAME.CNN_GRU,
        ]

        lstm_list = [
            constants.MODEL_NAME.DEEP_BILSTM,
            constants.MODEL_NAME.CNN_BILSTM,
            constants.MODEL_NAME.DEEP_LSTM,
            constants.MODEL_NAME.CNN_LSTM,
        ]

        self.direction = 'bidirectional' if config[constants.CONF_TAGS.MODEL] in bi_list else 'unidirectional'
        convolutional = config[constants.CONF_TAGS.MODEL] in conv_list
        self.kernel = 'LSTM' if config[constants.CONF_TAGS.MODEL] in lstm_list else 'GRU'

        if constants.CONF_TAGS.INIT_NPROJ in config:
            init_nproj = config[constants.CONF_TAGS.INIT_NPROJ]
        else:
            init_nproj = 0

        if constants.CONF_TAGS.FINAL_NPROJ in config:
            finalfeatproj = config[constants.CONF_TAGS.FINAL_NPROJ]
        else:
            finalfeatproj = 0

        self.batch_norm = config[constants.CONF_TAGS.BATCH_NORM]
        lstm_type = config[constants.CONF_TAGS.LSTM_TYPE]

        try:
            featproj = config["feat_proj"]
        except:
            featproj = 0

        # build the graph
        self.feats = tf.placeholder(tf.float32, [None, None, self.nfeat], name = "feats")
        self.is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.is_training = train
        self.is_trainable_sat=True
        self.logits = []

        self.seq_len = self.length(self.feats)

        self.batch_size = tf.shape(self.feats)[0]

        # Keep the same variable name throughout
        outputs = self.feats

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:

            sat_dim = config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM]
            self.sat = tf.placeholder(tf.float32, [None, 1, sat_dim], name="sat")
            sat_t = tf.transpose(self.sat, (1, 0, 2), name="sat_transpose")

            outputs = self.my_sat_module(config, outputs, sat_t)

        if self.batch_norm:
            outputs = self.batch_norm_fn(outputs)

        if init_nproj > 0:
            if convolutional:
                outputs = tf.expand_dims(outputs, axis = 3)
                outputs = self.make_layer(outputs, init_nproj, "down")
                outputs = tf.reshape(outputs, [self.batch_size, -1, (self.nfeat // 2) * init_nproj])
                outputs = tf.layers.dense(outputs, self.nproj)
            else:
                outputs = tf.contrib.layers.fully_connected(
                    activation_fn = None,
                    inputs = outputs,
                    num_outputs = init_nproj,
                    scope = "init_projection",
                    biases_initializer = tf.contrib.layers.xavier_initializer(),
                )

        # Transpose to length-first for lstm/ctc
        outputs = tf.transpose(outputs, (1, 0, 2), name = "feat_transpose")

        if lstm_type == "cudnn":
            outputs = self.my_cudnn_lstm(outputs)
        elif lstm_type == "fuse":
            outputs = self.my_fuse_block_lstm(outputs, "fuse_lstm")
        else:
            outputs = self.my_native_lstm(outputs, "native_lstm")


        if finalfeatproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = finalfeatproj,
                scope = "final_projection", biases_initializer = tf.contrib.layers.xavier_initializer())

        print(80 * "-")
        print("preparing model variables...")
        print(80 * "-")

        for language_id, language_target_dict in language_scheme.items():
            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    if len(language_scheme.items()) > 1:
                        scope="output_fc_"+language_id
                    if len(language_target_dict.items()) > 1:
                        scope="output_fc_"+target_id

                    if loss_fn == 'eemmi':
                        num_targets -= 1
                    
                    logit = tf.contrib.layers.fully_connected(
                        activation_fn      = None,
                        inputs             = outputs,
                        num_outputs        = num_targets,
                        scope              = "output_fc",
                        biases_initializer = tf.contrib.layers.xavier_initializer(),
                        trainable          = self.is_trainable_sat,
                    )

                    if self.batch_norm:
                        logit = self.batch_norm(logit, scope = "output_fc_bn")

                    #storing outputs
                    #tran_logit = tf.transpose(self.logit, (1, 0, 2))#* self.temperature
                    self.logits.append(logit)

            print(80 * "-")

