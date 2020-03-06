import constants
import tensorflow as tf
import numpy as np

def read_mat(filename):
    mat = []
    with open(filename) as f:
        for line in f:
            mat.append(line.split())

    mat = np.array(mat, np.float32)
    
    # Smooth and normalize language transition matrix
    mat = (mat + 1) / np.expand_dims(np.sum(mat, axis = 1) + mat.shape[1], axis = 1)

    return mat

class Trainer:

    def __init__(self, config, model, teacher_model = None, var_list = None):
        l2 = config[constants.CONF_TAGS.L2]
        clip = config[constants.CONF_TAGS.CLIP]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        loss_fn = config[constants.CONF_TAGS.LOSS]
        grad_opt = config[constants.CONF_TAGS.GRAD_OPT]

        # build the graph
        self.lr_rate = tf.placeholder(tf.float32, name = "learning_rate")[0]
        self.is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.is_training = var_list is not None
        self.opt = []

        if constants.CONFIG_TAGS_TEST.VERIFY_FILE in config:
            self.verify = tf.placeholder(tf.float32, [None], name = 'verify')

        self.labels=[]
        self.priors=[]
        self.alt = []

        #creating enough placeholders for out graph
        for language_id, target_scheme in language_scheme.items():
            for target_id, _ in target_scheme.items():
                if self.is_training or constants.CONFIG_TAGS_TEST.COMPUTE_TER in config:
                    self.labels.append(tf.sparse_placeholder(tf.int32))

                self.priors.append(tf.placeholder(tf.float32))
                if config[constants.CONF_TAGS.COMPUTE_ACC]:
                    for i in range(config[constants.CONFIG_TAGS_TEST.ALTERNATIVES]):
                        self.alt.append(tf.sparse_placeholder(tf.int32))


        #optional outputs for test
        self.ters = []
        self.costs = []
        self.debug_costs = []
        self.acc = []
        self.verify_acc = []

        #mantadory outpus for test
        self.softmax_probs = []
        self.log_softmax_probs = []
        self.log_likelihoods = []
        self.logits = []
        self.decodes = []

        self.framecount = []
        self.peakcount = []
        self.tdloss = []
        self.all_ters = []
        self.ones_ter = []
        self.ones_count = []
        self.nonones_ter = []
        self.nonones_count = []

        with tf.variable_scope("optimizer"):
            optimizer = None
            # TODO: cudnn only supports grad, add check for this
            if grad_opt == "grad":
                optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            elif grad_opt == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr_rate * 0.01)
            elif grad_opt == "momentum":
                optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)

        count=0

        print(80 * "-")
        print("preparing model variables...")
        print(80 * "-")

        for language_id, language_target_dict in language_scheme.items():

            tmp_cost, tmp_debug_cost, tmp_ter, tmp_logits, tmp_softmax_probs = [], [], [], [], []
            tmp_log_softmax_probs, tmp_log_likelihoods, tmp_decodes, tmp_acc, tmp_ver = [], [], [], [], []
            tmp_framecount, tmp_peakcount, tmp_tdloss = [], [], []
            one_ter, nonone_ter, one_count, nonone_count = [], [], [], []

            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    # Average over augmented data
                    #avg_logit = (model.logits[count][:,0::3]
                    #        + model.logits[count][:,1::3]
                    #        + model.logits[count][:,2::3]) / 3

                    # copy 3 times
                    #avg_logit = tf.tile(tf.expand_dims(avg_logit, axis = 2), [1, 1, 3, 1])
                    #avg_logit = tf.reshape(avg_logit, tf.shape(model.logits[count]))
                    #model.logits[count] = avg_logit

                    seq_len = tf.tile([tf.shape(model.logits[count])[0]], [tf.shape(model.logits[count])[1]])

                    if self.is_training or constants.CONFIG_TAGS_TEST.COMPUTE_TER in config \
                            and config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:

                        loss = 0

                        if 'eemmi' in loss_fn:
                            from eemmi.utils.loss_functions import mmi_loss
                            mat = read_mat("/scratch/tmp/plantingap/cslu/trans_prob.txt")
                            
                            num_targets -= 1

                            loss, _, _ = mmi_loss(
                                logits = model.logits[count],
                                sparse_labels = self.labels[count],
                                seq_lengths = seq_len,
                                num_labels = num_targets,
                                lang_transition_probs = mat,
                            )
                            #loss = tf.Print(loss, [loss])

                        if 'ctc' in loss_fn:
                            loss = tf.nn.ctc_loss(
                                labels = self.labels[count],
                                inputs = model.logits[count],
                                sequence_length = seq_len,
                            )
                        
                        if 'ts' in loss_fn and teacher_model is not None:

                            if 'lag' in loss_fn or 'avg' in loss_fn:

                                #ts_type = 'lag' if 'lag' in loss_fn else 'avg'
                                print(loss_fn)

                                import re
                                m = re.search(r'(-?\d)(lag|avg)(-?\d)', loss_fn)
                                
                                if m:
                                    start, ts_type, end = int(m.group(1)), m.group(2), int(m.group(3))
                                else:
                                    raise ValueError("Improperly formatted loss: " + loss_fn)

                                #start = int(loss_fn[loss_fn.find(ts_type)-1])
                                #end = int(loss_fn[loss_fn.find(ts_type)+3])

                                min_loss = tf.zeros(shape=(seq_len[0], tf.shape(seq_len)[0])) + 100000
                                avg_loss = tf.zeros(shape=(seq_len[0], tf.shape(seq_len)[0]))
                                for i in range(start, end+1):
                                    #shifted_idx = tf.roll(indexes, shift = i, axis = 0)
                                    #label = teacher_model.logits[count][shifted_idx]
                                    #loss = tf.losses.mean_squared_error(label, model.logits[count])
                                    shifted_logit = tf.roll(teacher_model.logits[count], shift = -i, axis = 0)
                                    loss_i = tf.losses.mean_squared_error(shifted_logit, model.logits[count])

                                    min_loss = tf.math.minimum(min_loss, loss_i)
                                    avg_loss += loss_i

                                if 'lag' == ts_type:
                                    loss += min_loss
                                else:
                                    loss += avg_loss / (end - start + 1)

                            else:

                                # If the teacher model is not being trained, this should have no effect.
                                #loss += tf.nn.ctc_loss(
                                #    labels = self.labels[count],
                                #    inputs = teacher_model.logits[count],
                                #    sequence_length = seq_len,
                                #)

                                # Match the teacher outputs (pre-softmax)
                                loss += tf.losses.mean_squared_error(teacher_model.logits[count], model.logits[count])

                        if 'td' in loss_fn:
                            length = tf.cast(tf.shape(model.logits[count])[0], dtype=tf.float32)
                            time_factor = tf.expand_dims(tf.range(length) / length, axis=1)
                            softmax_prob = tf.nn.softmax(model.logits[count])
                            post_sum = tf.reduce_sum(softmax_prob[:,:,:-1], axis=-1)
                            
                            # Find where posterior is > 0.1
                            nullify = tf.cast(tf.greater(post_sum, 0.1), dtype=tf.float32)

                            delay = tf.multiply(time_factor, nullify)
                            loss += tf.reduce_mean(tf.reduce_sum(delay, axis=0) / (tf.reduce_sum(nullify, axis=0) + 0.001))

                        if 'align' in loss_fn:
                            posterior = tf.nn.softmax(model.logits[count])
                            post_sum = tf.reduce_sum(posterior[:,:,:-1], axis=-1)
                            post_eps = posterior[:,:,-1]

                            #noneps = model.logits[count][:,:,:-1]
                            #eps = tf.expand_dims(model.logits[count][:,:,-1], axis = 2)

                            frame_energy = tf.reduce_sum(model.feats, axis=-1)
                            sum_mult = tf.less(frame_energy, tf.reduce_mean(frame_energy, axis=1, keepdims=True))
                            eps_mult = tf.logical_not(sum_mult)

                            sum_mult = tf.transpose(tf.cast(sum_mult, dtype=tf.float32))
                            eps_mult = tf.transpose(tf.cast(eps_mult, dtype=tf.float32))

                            loss += tf.log(post_sum * sum_mult + post_eps * eps_mult + 1e-8)

                        posterior = tf.nn.softmax(model.logits[count])[:,:,:-1]
                        post_sum = tf.reduce_sum(posterior, axis = -1)
                        nullify = tf.cast(tf.greater(post_sum, 0.1), dtype=tf.float32)
                        tmp_framecount.append(tf.reduce_sum(nullify))

                        peaks = tf.cast(tf.greater(posterior, 0.1), dtype=tf.float32)
                        toggle = peaks - tf.roll(peaks, 1, 0)
                        tmp_peakcount.append(tf.reduce_sum(tf.abs(toggle)) / 2)

                        onsets = tf.reduce_sum(tf.cast(tf.greater(toggle, 0.0), tf.float32), axis = -1)
                        length = tf.cast(tf.shape(model.logits[count])[0], dtype=tf.float32)
                        time_factor = tf.expand_dims(tf.range(length), axis=1)
                        onset_time = tf.multiply(time_factor, onsets)
                        onset_count = tf.reduce_sum(onsets, axis=0, keepdims=True)
                        onset_count = tf.maximum(onset_count, 1.0)
                        avg_onset = tf.reduce_sum(onset_time / onset_count)
                        tmp_tdloss.append(avg_onset)


                        tmp_cost.append(loss)
                        tmp_debug_cost.append(tf.reduce_mean(loss))

                        decoded, log_prob = tf.nn.ctc_greedy_decoder(model.logits[count], seq_len)
                        #decoded, log_prob = tf.nn.ctc_beam_search_decoder(model.logits[count], seq_len, 10)
                        prediction = tf.cast(decoded[0], tf.int32)
                        dense_decoded = tf.sparse_tensor_to_dense(decoded[0],default_value=-1)
                        tmp_decodes.append(dense_decoded)
                        
                        ter = tf.edit_distance(prediction, self.labels[count], normalize = False)
                        tmp_ter.append(tf.reduce_sum(ter))

                        self.all_ters.append(ter)

                    #storing outputs
                    tran_logit = tf.transpose(model.logits[count], (1, 0, 2))# * self.temperature
                    tmp_logits.append(tran_logit)

                    softmax_prob = tf.nn.softmax(tran_logit, dim=-1, name=None)
                    tmp_softmax_probs.append(softmax_prob)

                    log_softmax_prob = tf.log(softmax_prob)
                    tmp_log_softmax_probs.append(log_softmax_prob)

                    log_likelihood = log_softmax_prob - tf.log(self.priors[count])
                    tmp_log_likelihoods.append(log_likelihood)


                    if config[constants.CONF_TAGS.COMPUTE_ACC]:
                        accuracy = tf.greater(ter, -1)
                        for alternative in self.alt:
                            alt_ter = tf.edit_distance(prediction, alternative, normalize = False)
                            accuracy = tf.logical_and(accuracy, tf.greater_equal(alt_ter, ter))
                        tmp_acc.append(tf.reduce_sum(tf.cast(accuracy, tf.float32)))

                    if constants.CONFIG_TAGS_TEST.VERIFY_FILE in config:
                        guess = tf.cast(tf.greater(ter, 1), tf.int32)
                        label = tf.cast(tf.greater(self.verify, 0), tf.int32)
                        TP = tf.count_nonzero(guess * label)
                        TN = tf.count_nonzero((guess - 1) * (label - 1))
                        FP = tf.count_nonzero(guess * (label - 1))
                        FN = tf.count_nonzero((guess - 1) * label)

                        tmp_ver.append([TP, TN, FP, FN])

                        ter = tf.cast(ter, tf.int32)
                        nonone_ter.append(tf.reduce_sum(ter * label))

                        onelabel = tf.cast(tf.equal(self.verify, 0), tf.int32)
                        one_ter.append(tf.reduce_sum(ter * onelabel))


                    count=count+1

            #preparing variables to optimize
            #if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
            #    == constants.SAT_SATGES.TRAIN_SAT:
            #    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=constants.SCOPES.SPEAKER_ADAPTAION)
            #else:
            #    if len(language_scheme.items()) > 1:
            #        var_list = self.get_variables_by_lan(language_id)
            #    else:
            #        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


            if self.is_training:
                print(80 * "-")
                if len(language_scheme.items()) > 1:
                    print("for language: "+language_id)
                print("following variables will be optimized: ")
                print(80 * "-")
                for var in var_list:
                    print(var)
                print(80 * "-")

                with tf.variable_scope("loss"):
                    regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])

                #reduce the mean of all targets of current language(language_id)
                tmp_cost = tf.reduce_mean(tmp_cost) + l2 * regularized_loss
            else:
                print(80 * "-")
                print("testing... no variables will be optimized.")
                print(80 * "-")


            self.debug_costs.append(tmp_debug_cost)
            self.costs.append(tmp_cost)
            self.ters.append(tmp_ter)
            self.acc.append(tmp_acc)
            self.verify_acc.append(tmp_ver)
            self.logits.append(tmp_logits)
            self.softmax_probs.append(tmp_softmax_probs)
            self.log_softmax_probs.append(tmp_log_softmax_probs)
            self.log_likelihoods.append(tmp_log_likelihoods)
            self.decodes.append(tmp_decodes)

            self.framecount.append(tmp_framecount)
            self.peakcount.append(tmp_peakcount)
            self.tdloss.append(tmp_tdloss)

            self.ones_ter.append(one_ter)
            self.ones_count.append(one_count)
            self.nonones_ter.append(nonone_ter)
            self.nonones_count.append(nonone_count)

            if self.is_training:
                gvs = optimizer.compute_gradients(tmp_cost, var_list=var_list)

                #capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]
                capped_gvs = [(tf.clip_by_norm(grad, clip), var) for grad, var in gvs]

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

