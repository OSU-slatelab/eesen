import constants
import tensorflow as tf
from utils.fileutils import debug
import sys
from eemmi.utils.loss_functions import mmi_loss
import numpy as np

def read_mat(filename):
    mat = []
    with open(filename) as f:
        for line in f:
            mat.append(line.split())

    mat = np.array(mat, np.float32)
    mat = (mat + 1) / (np.sum(mat, axis = 0) + 42)

    return mat

class Resnet:

    #def length(self, sequence):
    #    with tf.variable_scope("seq_len"):
    #        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    #        length = tf.reduce_sum(used, axis=1)
    #        length = tf.cast(length, tf.int32)
    #    return length

    def my_sat_layers(self, num_sat_layers, adapt_dim, nfeat, outputs):

        for i in range(num_sat_layers-1):
            with tf.variable_scope("layer%d" % i):
                outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = adapt_dim)

        with tf.variable_scope("last_sat_layer"):
            outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = nfeat)

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

                print("this sat type ("+str(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE])+") was not contemplates")
                print(debug.get_debug_info())
                print("exiting...")
                sys.exit()

    def make_model(self, inputs, nfeat):

        block = tf.expand_dims(inputs, axis=1)

        filters = [64, 128, 256]
        for i, f in enumerate(filters):
            with tf.variable_scope("block{}".format(i)):
                block = self.make_block(block, f)

        # [batch_size, filters[-1], length // 2^len(filters), height // 2^len(filters)]
        shape = tf.shape(block)
        block = tf.transpose(block, [2, 0, 1, 3])

        fc = tf.reshape(block, [shape[2], shape[0], filters[-1] * (nfeat // 2 ** len(filters))])
        fc = tf.layers.dense(fc, 1024, tf.nn.relu)

        return fc

    def make_block(self, inputs, filters):
        with tf.variable_scope("downsample"):
            downsample = self.make_layer(inputs, filters, downsample = True)
        with tf.variable_scope("conv1"):
            conv1 = self.make_layer(tf.nn.relu(downsample), filters)
        with tf.variable_scope("conv2"):
            conv2 = self.make_layer(conv1, filters, dropout = False)
        return downsample + conv2

    def make_layer(self, inputs, filters, downsample = False, dropout = True):
        layer = tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = 3 if downsample else (1, 5),
            strides = (1, 2) if downsample else 1,
            padding = 'same',
            data_format = 'channels_first',
            activation = tf.nn.relu if not downsample else None,
        )

        if dropout:
            layer = tf.layers.dropout(
                inputs = layer,
                rate = 0.3,
                training = self.is_training_ph,
                noise_shape = [self.batch_size, filters, 1, 1],
            )

        return layer

    def __init__(self, config):

        nfeat = config[constants.CONF_TAGS.INPUT_FEATS_DIM]

        nhidden = config[constants.CONF_TAGS.NHIDDEN]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        l2 = config[constants.CONF_TAGS.L2]
        nlayer = config[constants.CONF_TAGS.NLAYERS]
        clip = config[constants.CONF_TAGS.CLIP]
        nproj = config[constants.CONF_TAGS.NPROJ]
        dropout = config[constants.CONF_TAGS.DROPOUT]

        if constants.CONF_TAGS.INIT_NPROJ in config:
            init_nproj = config[constants.CONF_TAGS.INIT_NPROJ]
        else:
            init_nproj = 0

        if constants.CONF_TAGS.FINAL_NPROJ in config:
            finalfeatproj = config[constants.CONF_TAGS.FINAL_NPROJ]
        else:
            finalfeatproj = 0

        batch_norm = config[constants.CONF_TAGS.BATCH_NORM]
        lstm_type = config[constants.CONF_TAGS.LSTM_TYPE]
        grad_opt = config[constants.CONF_TAGS.GRAD_OPT]

        if constants.CONFIG_TAGS_TEST in config:
            self.is_training = False
        else:
            self.is_training = True

        self.is_trainable_sat=True

        try:
            featproj = config["feat_proj"]
        except:
            featproj = 0

        # build the graph
        self.lr_rate = tf.placeholder(tf.float32, name = "learning_rate")[0]
        self.feats = tf.placeholder(tf.float32, [None, None, nfeat], name = "feats")
        self.temperature = tf.placeholder(tf.float32, name = "temperature")
        self.is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.opt = []

        self.labels=[]
        self.priors=[]

        #optional outputs for test
        self.ters = []
        self.costs = []
        self.debug_costs = []
        self.acc = [[tf.constant(0)]]
        self.count = [[tf.constant(0)]]

        #mantadory outpus for test
        self.softmax_probs = []
        self.log_softmax_probs = []
        self.log_likelihoods = []
        self.seq_len = []
        self.logits = []
        self.decodes = []

        #creating enough placeholders for out graph
        for language_id, target_scheme in language_scheme.items():
            for target_id, _ in target_scheme.items():
                self.labels.append(tf.sparse_placeholder(tf.int32))
                self.priors.append(tf.placeholder(tf.float32))

        #self.seq_len = self.length(self.feats)
        self.batch_size = tf.shape(self.feats)[0]
        #outputs = tf.transpose(self.feats, (1, 0, 2), name = "feat_transpose")
        #batch_size, self.seq_len, feat_size = tf.shape(self.feats)
        outputs = self.feats

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            print("running sat " * 100)
            shape = [None, 1, config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM]]
            self.sat = tf.placeholder(tf.float32, shape, name="sat")
            #sat_t = tf.transpose(self.sat, (1, 0, 2), name="sat_transpose")

            outputs = self.my_sat_module(config, outputs, sat_t)

        if batch_norm:
            outputs = tf.contrib.layers.batch_norm(
                outputs,
                scope = "bn",
                center = True,
                scale = True,
                decay = 0.9,
                is_training = self.is_training_ph,
                updates_collections = None,
            )


        #if init_nproj > 0:
        #    outputs = tf.contrib.layers.fully_connected(
        #        activation_fn = None,
        #        inputs = outputs,
        #        num_outputs = init_nproj,
        #        scope = "init_projection",
        #        biases_initializer = tf.contrib.layers.xavier_initializer(),
        #    )

        outputs = self.make_model(outputs, nfeat)
        self.seq_len = tf.tile([tf.shape(outputs)[0]], [tf.shape(outputs)[1]])

        #if finalfeatproj > 0:
        #    outputs = tf.contrib.layers.fully_connected(
        #        activation_fn = None,
        #        inputs = outputs,
        #        num_outputs = finalfeatproj,
        #        scope = "final_projection",
        #        biases_initializer = tf.contrib.layers.xavier_initializer(),
        #    )

        with tf.variable_scope("optimizer"):
            #if grad_opt == "grad":
            #    optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            #elif grad_opt == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr_rate * 0.01)
            #elif grad_opt == "momentum":
            #    optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)
            #else:
            #    raise ValueError("invalid optimizer")

        count=0

        print(80 * "-")
        print("preparing model variables...")
        print(80 * "-")


        for language_id, language_target_dict in language_scheme.items():


            tmp_cost, tmp_debug_cost, tmp_ter, tmp_logits, tmp_softmax_probs, tmp_log_softmax_probs, tmp_log_likelihoods, tmp_decodes = [], [], [], [], [], [], [], []

            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    scope = "output_fc"

                    if len(language_scheme.items()) > 1:
                        scope=scope+"_"+language_id
                    if len(language_target_dict.items()) > 1:
                        scope=scope+"_"+target_id

                    logit = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs,
                                                              num_outputs=num_targets-1,
                                                              scope = scope,
                                                              biases_initializer = tf.contrib.layers.xavier_initializer(),
                                                              trainable=self.is_trainable_sat)
                    if batch_norm:
                        logit = tf.contrib.layers.batch_norm(logit, scope = scope+"_bn", center=True, scale=True, decay=0.9,
                                                             is_training=self.is_training_ph, updates_collections=None)

                    mat = read_mat("/scratch/tmp/plantingap/cslu/trans_prob.txt")
                    loss, _, _ = mmi_loss(logits=logit, sparse_labels=self.labels[count], seq_lengths=self.seq_len,
                        num_labels=num_targets, lang_transition_probs=mat)
                    
                    #loss = tf.nn.ctc_loss(labels=self.labels[count], inputs=logit, sequence_length=self.seq_len)
                    #loss = tf.Print(loss, [loss])
                    
                    tmp_cost.append(loss)
                    tmp_debug_cost.append(tf.reduce_mean(loss))

                    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logit, self.seq_len)
                    #decoded, log_prob = tf.nn.ctc_greedy_decoder(logit, self.seq_len)
                    #dense_decoded=tf.sparse_tensor_to_dense(decoded[0],default_value=-1)
                    #tmp_decodes.append(dense_decoded)
                    
                    #ter = tf.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels[count], normalize = False))
                    #tmp_ter.append(ter)
                    tmp_ter.append(tf.reduce_mean(loss))

                    #storing outputs
                    tran_logit = tf.transpose(logit, (1, 0, 2)) * self.temperature
                    tmp_logits.append(tran_logit)

                    softmax_prob = tf.nn.softmax(tran_logit, dim=-1, name=None)
                    tmp_softmax_probs.append(softmax_prob)

                    log_softmax_prob = tf.log(softmax_prob)
                    tmp_log_softmax_probs.append(log_softmax_prob)

                    log_likelihood = log_softmax_prob - tf.log(self.priors[count])
                    tmp_log_likelihoods.append(log_likelihood)

                    count=count+1

            #preparing variables to optimize
            if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                == constants.SAT_SATGES.TRAIN_SAT:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=constants.SCOPES.SPEAKER_ADAPTAION)
            else:
                if len(language_scheme.items()) > 1:
                    var_list = self.get_variables_by_lan(language_id)
                else:
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


            if self.is_training:
                print(80 * "-")
                if len(language_scheme.items()) > 1:
                    print("for language: "+language_id)
                print("following variables will be optimized: ")
                print(80 * "-")
                for var in var_list:
                    print(var)
                print(80 * "-")
            else:
                print(80 * "-")
                print("testing... no variables will be optimized.")
                print(80 * "-")

            with tf.variable_scope("loss"):
                regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])

            #reduce the mean of all targets of current language(language_id)
            tmp_cost = tf.reduce_mean(tmp_cost) + l2 * regularized_loss


            self.debug_costs.append(tmp_debug_cost)
            self.costs.append(tmp_cost)
            self.ters.append(tmp_ter)
            self.logits.append(tmp_logits)
            self.softmax_probs.append(tmp_softmax_probs)
            self.log_softmax_probs.append(tmp_log_softmax_probs)
            self.log_likelihoods.append(tmp_log_likelihoods)
            self.decodes.append(tmp_decodes)

            gvs = optimizer.compute_gradients(tmp_cost, var_list=var_list)

            capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]

            #at end  of the day we will decide whch optimizer to call:
            #each one will optimize over all targets of the selected language (correct idx)
            self.opt.append(optimizer.apply_gradients(capped_gvs))

            print(80 * "-")

    def get_variables_by_lan(self, current_name):

        train_vars=[]
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if "output_fc" not in var.name:
                train_vars.append(var)
            elif current_name in var.name:
                train_vars.append(var)

        return train_vars

