import constants
import tensorflow as tf
from utils.fileutils import debug
import sys


class ResnetBilstm:

    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length


    def my_cudnn_lstm(self, outputs, nlayer, nhidden, nfeat, nproj, batch_norm, dropout):
        """
        outputs: time, batch_size, feat_dim
        """
        if (nlayer == 0):
            sys.exit()
            
        if (nproj > 0):
            ninput = nfeat
            for i in range(nlayer):
                with tf.variable_scope("layer%d" % i):

                    cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(
                        num_layers         = 1,
                        num_units          = nhidden,
                        input_mode         = 'linear_input',
                        direction          = 'bidirectional',
                        dropout            = dropout,
                        kernel_initializer = tf.contrib.layers.xavier_initializer(),
                        bias_initializer   = tf.contrib.layers.xavier_initializer(),
                    )


                    outputs, _output_h = cudnn_model(outputs, None, True)


                    #biases initialized in 0 (default)
                    #weights initialized with xavier (default)
                    outputs = tf.contrib.layers.fully_connected(
                        activation_fn=None,
                        inputs=outputs,
                        num_outputs=nproj,
                        scope="intermediate_projection")


                    if batch_norm:
                        outputs = tf.contrib.layers.batch_norm(outputs,
                        scope = "bn", center=True, scale=True, decay=0.9,
                        is_training=self.is_training_ph, updates_collections=None)

        else:
            cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers         = nlayer,
                num_units          = nhidden,
                input_mode         = 'linear_input',
                direction          = 'bidirectional',
                dropout            = dropout,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                bias_initializer   = tf.contrib.layers.xavier_initializer(),
            )


            # arguments are: inputs, initial_state,
            outputs, _output_h = cudnn_model(outputs, None, True)
            #outputs_i, _output_h_i = cudnn_model(outputs, None, False)

            if batch_norm:
                outputs = tf.contrib.layers.batch_norm(outputs,
                    scope = "bn", center=True, scale=True, decay=0.9,
                    is_training=self.is_training_ph, updates_collections=None)

        return outputs

    def conv_layer(self, inputs, channels, downsample = False, dropout = True):
        
        conv = tf.layers.conv2d(
            inputs = inputs,
            filters = channels,
            kernel_size = 3,
            strides = (1,2) if downsample else 1,
            data_format = 'channels_first',
            padding = 'same',
            activation = tf.nn.relu if not downsample else None,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
        )

        if dropout:
            conv = tf.layers.dropout(
                inputs = conv,
                rate = 0.3,
                training = self.is_training_ph,
                noise_shape = [self.batch_size, channels, 1, 1],
            )

        return conv
    
    def resnet_block(self, inputs, channels):
        
        with tf.variable_scope('downsample'):
            downsampled = self.conv_layer(inputs, channels, downsample = True)

        with tf.variable_scope('conv1'):
            conv1 = self.conv_layer(tf.nn.relu(downsampled), channels)

        with tf.variable_scope('conv2'):
            conv2 = self.conv_layer(conv1, channels, dropout = False)

        return downsampled + conv2

    def __init__(self, config):

        nfeat = config[constants.CONF_TAGS.INPUT_FEATS_DIM]

        nhidden = config[constants.CONF_TAGS.NHIDDEN]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        l2 = config[constants.CONF_TAGS.L2]
        nlayer = config[constants.CONF_TAGS.NLAYERS]
        clip = config[constants.CONF_TAGS.CLIP]
        nproj = config[constants.CONF_TAGS.NPROJ]
        dropout = config[constants.CONF_TAGS.DROPOUT]
        clip_norm = config[constants.CONF_TAGS.CLIP_NORM]
        kl_weight = config[constants.CONF_TAGS.KL_WEIGHT]

        batch_norm = config[constants.CONF_TAGS.BATCH_NORM]
        lstm_type = config[constants.CONF_TAGS.LSTM_TYPE]
        grad_opt = config[constants.CONF_TAGS.GRAD_OPT]

        tf.set_random_seed(config[constants.CONF_TAGS.RANDOM_SEED])


        if constants.CONF_TAGS.INIT_NPROJ in config:
            init_nproj = config[constants.CONF_TAGS.INIT_NPROJ]
        else:
            init_nproj = 0

        if constants.CONF_TAGS.FINAL_NPROJ in config:
            finalfeatproj = config[constants.CONF_TAGS.FINAL_NPROJ]
        else:
            finalfeatproj = 0

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
        self.alt=[]

        #optional outputs for test
        self.acc = []
        self.ters = []
        self.costs = []
        self.debug_costs = []

        #mantadory outpus for test
        self.softmax_probs = []
        self.log_softmax_probs = []
        self.log_likelihoods = []
        self.logits = []

        #creating enough placeholders for out graph
        for language_id, target_scheme in language_scheme.items():
            for target_id, _ in target_scheme.items():
                self.labels.append(tf.sparse_placeholder(tf.int32))
                self.priors.append(tf.placeholder(tf.float32))
                if config[constants.CONF_TAGS.COMPUTE_ACC]:
                    for i in range(config[constants.CONFIG_TAGS_TEST.ALTERNATIVES]):
                        self.alt.append(tf.sparse_placeholder(tf.int32))

        #self.seq_len = self.length(self.feats)

        self.batch_size = tf.shape(self.feats)[0]
        
        # Resnet
        channels = 64
        resnet_inputs = tf.expand_dims(self.feats, axis=1)
        outputs = self.resnet_block(resnet_inputs, channels = channels)

        # Prepare for LSTM
        outputs = tf.transpose(outputs, [2, 0, 1, 3])
        #print_op = tf.print(tf.shape(outputs))
        #with tf.control_dependencies([print_op]):
        outputs = tf.reshape(outputs, [-1, self.batch_size, channels * (nfeat // 2)])
        self.seq_len = tf.tile([tf.shape(outputs)[0]], [self.batch_size])

        if batch_norm:
            outputs = tf.contrib.layers.batch_norm(outputs, scope = "bn", center=True, scale=True,
                    decay=0.9, is_training=self.is_training_ph, updates_collections=None)

        if init_nproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = init_nproj,
                scope = "init_projection")

        with tf.variable_scope('cudnn_lstm'):
            outputs = self.my_cudnn_lstm(outputs, nlayer, nhidden, nfeat, nproj, batch_norm, dropout)

        if finalfeatproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = finalfeatproj,
                scope = "final_projection")

        with tf.variable_scope("optimizer"):
            if grad_opt == "grad":
                optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            elif grad_opt == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr_rate)
            elif grad_opt == "momentum":
                optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)
            else:
                raise ValueError("optimizer must be one of 'grad', 'adam', or 'momentum'")

        count=0

        print(80 * "-")
        print("preparing model variables...")
        print(80 * "-")


        for language_id, language_target_dict in language_scheme.items():


            tmp_ctc_cost, tmp_debug_cost, tmp_ter, tmp_logits, tmp_softmax_probs = [], [], [], [], []
            tmp_log_softmax_probs, tmp_log_likelihoods, tmp_kl_cost, tmp_acc = [], [], [], []

            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    scope = "output_fc"

                    if len(language_scheme.items()) > 1:
                        scope=scope+"_"+language_id
                    if len(language_target_dict.items()) > 1:
                        scope=scope+"_"+target_id

                    if batch_norm:
                        outputs = tf.contrib.layers.batch_norm(
                            outputs,
                            scope               = scope+"_bn",
                            center              = True,
                            scale               = True,
                            decay               = 0.9,
                            is_training         = self.is_training_ph,
                            updates_collections = None,
                        )


                    logit = tf.contrib.layers.fully_connected(
                        activation_fn      = None,
                        inputs             = outputs,
                        num_outputs        = num_targets,
                        scope              = scope,
                        biases_initializer = tf.contrib.layers.xavier_initializer(),
                        trainable          = self.is_trainable_sat,
                    )

                    #######
                    #here logits: time, batch, num_classes+1
                    #######

                    ctc_loss = tf.nn.ctc_loss(labels=self.labels[count], inputs=logit, sequence_length=self.seq_len)

                    tmp_ctc_cost.append(ctc_loss)
                    tmp_debug_cost.append(tf.reduce_mean(ctc_loss))

                    decoded, log_prob = tf.nn.ctc_greedy_decoder(logit, self.seq_len)
                    prediction = tf.cast(decoded[0], tf.int32)

                    ter = tf.edit_distance(prediction, self.labels[count], normalize = False)
                    tmp_ter.append(tf.reduce_sum(ter))

                    #storing outputs
                    tran_logit = tf.transpose(logit, (1, 0, 2)) * self.temperature
                    tmp_logits.append(tran_logit)

                    softmax_prob = tf.nn.softmax(tran_logit, dim=-1, name=None)
                    tmp_softmax_probs.append(softmax_prob)

                    log_softmax_prob = tf.log(softmax_prob)
                    tmp_log_softmax_probs.append(log_softmax_prob)

                    log_likelihood = log_softmax_prob - tf.log(self.priors[count])
                    tmp_log_likelihoods.append(log_likelihood)                

                    accuracy = tf.math.greater(ter, -1)
                    for alternative in self.alt:
                        alt_ter = tf.edit_distance(prediction, alternative, normalize = False)
                        accuracy = tf.math.logical_and(accuracy, tf.math.greater_equal(alt_ter, ter))

                    tmp_acc.append(tf.reduce_sum(tf.cast(accuracy, tf.float32)))

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
            tmp_ctc_cost = tf.reduce_mean(tmp_ctc_cost) + l2 * regularized_loss

            self.debug_costs.append(tmp_debug_cost)
            self.costs.append(tmp_ctc_cost)
            self.ters.append(tmp_ter)
            self.acc.append(tmp_acc)
            self.logits.append(tmp_logits)
            self.softmax_probs.append(tmp_softmax_probs)

            self.log_softmax_probs.append(tmp_log_softmax_probs)
            self.log_likelihoods.append(tmp_log_likelihoods)

            gvs = optimizer.compute_gradients(tmp_ctc_cost, var_list=var_list)

            if clip_norm:
                capped_gvs = [(tf.clip_by_norm(grad, clip), var) for grad, var in gvs]
            else:
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

