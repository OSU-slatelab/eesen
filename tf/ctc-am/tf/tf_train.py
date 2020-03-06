from multiprocessing import Process, Queue
import sys
import tensorflow as tf
#if tf.__version__ == "1.6.0":
#from models_16.model_factory import create_model
#else:
from models.model_factory import create_model
from trainer import Trainer

import os, re, time, random
import tensorflow as tf
import pdb
import constants
import numpy as np

from reader.reader_queue import run_reader_queue
from random import randint
from collections import deque
from utils.fileutils.kaldi import writeArk, writeScp
from lrscheduler.lrscheduler_factory import create_lrscheduler

class Train():

    def __init__(self, config):

        self.__config = config

        self.__sess = tf.Session()

        self.__teacher_model = None
        teacher_vars = []
        self.train_teacher = False
        if self.__config[constants.CONF_TAGS.TEACHER_CONFIG]:

            # Train teacher model jointly if there's no pre-trained weights
            self.train_teacher = self.__config[constants.CONF_TAGS.TEACHER_WEIGHTS] == ""
            
            import pickle
            teacher_config = pickle.load(open(self.__config[constants.CONF_TAGS.TEACHER_CONFIG], 'rb'))

            # Load teacher config and create a model from it
            self.__teacher_model = create_model(teacher_config, 'teacher/student', train = self.train_teacher)
            #self.__teacher_model = create_model(teacher_config, 'teacher', train = self.train_teacher)
            teacher_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='teacher')

            # Add teacher to variable scope and load weights
            if not self.train_teacher:
                teacher_saver = tf.train.Saver({var.op.name[8:]: var for var in teacher_vars})
                teacher_saver.restore(self.__sess, self.__config[constants.CONF_TAGS.TEACHER_WEIGHTS])

        self.__model = create_model(config, 'student', self.__teacher_model, train = True)
        student_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student')

        self.training_vars = student_vars
        if self.train_teacher:
            self.training_vars += teacher_vars

        with tf.variable_scope('trainer'):
            self.__trainer = Trainer(config, self.__model, self.__teacher_model, self.training_vars)

        trainer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainer')
        self.training_vars += trainer_vars

        self.max_targets_layers = 0
        self.__ter_buffer = [float('inf'), float('inf')]

        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if self.max_targets_layers < len(target_scheme):
                self.max_targets_layers = len(target_scheme)

    def train_impl(self, data):

        #set random seed so that lm_models can be reproduced
        tf.set_random_seed(self.__config[constants.CONF_TAGS.RANDOM_SEED])
        random.seed(self.__config[constants.CONF_TAGS.RANDOM_SEED])

        tr_data, cv_data = data

        #construct the __model according to __config
        if not os.path.exists(self.__config[constants.CONF_TAGS.MODEL_DIR]):
            os.makedirs(self.__config[constants.CONF_TAGS.MODEL_DIR])

        #initialize variables of our model
        self.__sess.run(tf.initializers.variables(self.training_vars))
        self.__lrscheduler=create_lrscheduler(self.__config)

        # restore a training - note this will now update lrscheduler as well
        saver, alpha, best_avg_ters = self.__restore_weights()

        #initialize counters
        best_epoch = alpha

        epoch, lr_rate = self.__lrscheduler.initialize_training()

        # Main training loop. LRscheduler will set should_stop flag
        should_stop = False
        while not should_stop:

            # log start
            self.__info("Epoch %d starting, learning rate: %.4g" % (epoch, lr_rate))

            # start timer
            tic = time.time()

            # training
            tr_stats = self.__run_epoch(epoch, tr_data, lr_rate)

            if self.__config[constants.CONF_TAGS.STORE_MODEL]:
                saver.save(self.__sess, "%s/epoch%02d.ckpt" % (self.__config[constants.CONF_TAGS.MODEL_DIR], epoch))

            # evaluate on validation
            cv_stats = self.__run_epoch(epoch, cv_data)

            # update lr_rate if needed
            new_epoch, new_lr_rate, should_stop, restore = self.__lrscheduler.update_lr_rate(cv_stats)

            # generate logs after lrscheduler update so that the log can put out lrscheduler info
            self.__generate_logs(cv_stats, tr_stats, epoch, lr_rate, tic)
            
            # check restoring before stopping in case last model was suboptimal
            if restore is not None:
                self.__restore_epoch(saver, restore)

            # Prepare for next epoch
            epoch = new_epoch
            lr_rate = new_lr_rate
            self.__update_sets(tr_data)

        # save final model after all iterations
        if self.__config[constants.CONF_TAGS.STORE_MODEL]:
                saver.save(self.__sess, "%s/final.ckpt" % (self.__config[constants.CONF_TAGS.MODEL_DIR]))

    def __restore_epoch(self, saver, best_epoch):
        epoch_name = "/epoch%02d.ckpt" % (best_epoch)
        best_epoch_path = self.__config[constants.CONF_TAGS.MODEL_DIR] + epoch_name

        if os.path.isfile(best_epoch_path+".index"):
            print("restoring model from epoch {}".format(best_epoch))
            saver.restore(self.__sess, "%s/epoch%02d.ckpt" % (self.__config["model_dir"], best_epoch))
        else:
            print("epoch {} NOT found. restoring can not be done. ({})".format(
                best_epoch,
                best_epoch_path,
            ))

    def __update_sets(self, m_tr):

        #print(80 * "-")
        #print("checking update of epoch...")
        #this is fundamentally wrong
        if self.__check_needed_mix_augmentation(m_tr['x']):
            dic_sources={}

            #randomizing over all lanaguages (aka geting a number for each language)
            for language_id, lan_aug_folders in m_tr['x'].get_language_scheme().items():
                new_src = randint(0, len(lan_aug_folders)-1)
                dic_sources[language_id] = new_src

            print(80 * "-")
            print("changing tr_x sources...")
            print(80 * "-")
            #get a random folder of all previously provided
            m_tr['x'].change_source(dic_sources)
            for key in m_tr:
                if key != 'x':
                    m_tr[key].update_batches_id(m_tr['x'].get_batches_id())

    def __check_needed_mix_augmentation(self, tr_x):
        for target_id, augment_dirs in tr_x.get_language_scheme().items():
            if len(augment_dirs) > 1:
                return True
        return False

    def __run_epoch(self, epoch, data, learn_rate = None):

        # init data_queue
        data_queue = Queue(self.__config["batch_size"])

        #initializing samples, steps and cost counters
        batch_counter = 0

        #initializinzing dictionaries that will count
        stats = {}

        #TODO change all iteritems for iter for python 3.0
        #TODO try to do an lm_utils for this kind of functions
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            stats[language_id] = {}
            stats[language_id]['n'] = 0

            for target_id, _ in target_scheme.items():
                stats[language_id][target_id] = {
                    'label' : 0,
                    #'acc'   : 0,
                    'ter'   : 0,
                    'cost'  : 0,
                    #'count' : 0,
                }

        #kwargs = {
        #    'queue'      : data_queue,
        #    'reader_x'   : data['x'],
        #    'reader_y'   : data['y'],
        #    'do_shuf'    : self.__config[constants.CONF_TAGS.DO_SHUF],
        #    'is_debug'   : self.__config[constants.CONF_TAGS.DEBUG],
        #    'seed'       : self.__config[constants.CONF_TAGS.RANDOM_SEED] + epoch,
        #}

        #if 'sat' in data:
        #    kwargs['reader_sat'] = data['sat']
        #if 'z' in data:
        #    kwargs['reader_z'] = data['z']

        #p = Process(target = run_reader_queue, kwargs = kwargs)

        #start queue ...
        #p.start()

        #training starting...
        #while True:
        for idx in np.random.permutation(data['x'].get_num_batches()):

            #pop from queue
            #data = data_queue.get()
            batch_data = (data['x'].read(idx), data['y'].read(idx), None, None)

            #finish if there no more batches
            #if data is None:
            #    break

            batch = {}
            feed, batch['size'], index_correct_lan, _ = self.__prepare_feed(batch_data, learn_rate)

            #request_items = ['cost', 'ters', 'acc', 'count']
            request_items = ['cost', 'ters']
            ops = [
                self.__trainer.debug_costs[index_correct_lan],
                self.__trainer.ters[index_correct_lan],
                #self.__trainer.acc[index_correct_lan],
                #self.__trainer.count[index_correct_lan],
            ]

            if self.__config[constants.CONF_TAGS.DUMP_CV_FWD]:
                ops.append(self.__trainer.decodes[index_correct_lan])
                ops.append(self.__trainer.logits)
                ops.append(self.__trainer.seq_len)
                request_items += ['decodes', 'logits', 'seq_len']

            # If there's a learn_rate, we're training
            if learn_rate:
                ops += self.__trainer.opt[index_correct_lan],

            outputs = self.__sess.run(ops, feed)
            for i, item in enumerate(request_items):
                batch[item] = outputs[i]

            #updating values...
            self.__update_counters(stats, batch, batch_data[1])

            #print if in debug mode
            if self.__config[constants.CONF_TAGS.DEBUG] == True:
                self.__print_counts_debug(epoch, batch_counter, data['x'].get_num_batches(), batch, data_queue)
                batch_counter += 1
                for cur_id, cur_decode in zip(batch_id, batch_decodes[0]):
                    # add one to convert from tf (blank==last) back to our labeling scheme (blank==0)
                    decode_list=[str(i+1) for i in cur_decode if i>=0]
                    decode_string=' '.join(decode_list)
                    print('DECODE epoch %d:%s %s' % (epoch, cur_id, decode_string))

        #p.join()
        #p.terminate()

        #averaging counters
        self.__average_counters(stats)

        return stats

    def __create_result_containers(self, config):

        batches_id={}
        logits = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            batches_id[language_id] = []
            logits[language_id] = {}
            for target_id, _ in target_scheme.items():
                logits[language_id][target_id]=[]

        return logits, batches_id

    def __average_counters(self, stats):
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            for target_id in target_scheme:
                if stats[language_id]['n'] != 0:
                    stats[language_id][target_id]['cost'] /= float(stats[language_id]['n'])
                    #stats[language_id][target_id]['acc'] /= float(stats[language_id]['n'])
                    #stats[language_id][target_id]['count'] /= float(stats[language_id]['n'])
                if stats[language_id][target_id]['label'] != 0:
                    stats[language_id][target_id]['ter'] /= float(stats[language_id][target_id]['label'])

    def __update_probs_containers(self, config, batch, m_batches_id, m_logits):

        language_idx=0
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            m_batches_id[language_id] += batch['id']
            target_idx=0
            for target_id, num_targets in target_scheme.items():
                m_logits[language_id][target_id] += self.__mat2list(batch['logits'][language_idx][target_idx], batch['seq_len'])
                target_idx += 1
            language_idx += 1


    def __update_counters(self, stats, batch, ybatch):

        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for idx_lan, (language_id, target_scheme) in enumerate(self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items()):
            if ybatch[1] == language_id:
                for idx_tar, (target_id, _) in enumerate(target_scheme.items()):

                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    #stats[language_id][target_id]['acc'] += batch['acc'][idx_tar]
                    stats[language_id][target_id]['ter'] += batch['ters'][idx_tar]
                    #stats[language_id][target_id]['count'] += batch['count'][idx_tar]
                    stats[language_id][target_id]['label'] += self.__get_label_len(ybatch[0][language_id][target_id])
                    if batch['cost'][idx_tar] != float('Inf'):
                        stats[language_id][target_id]['cost'] += batch['cost'][idx_tar] * batch['size']

            stats[language_id]['n'] += batch['size']

    def __average_over_augmented_data(self, config, m_batches_id, m_logits):
        #new batch structure
        new_batch_id = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            new_batch_id[language_id] = {}

            for target_id, num_targets in target_scheme.items():
                S={}; P={}; O={}
                    #iterate over all utterances of a concrete language
                for utt_id, s in zip(m_batches_id[language_id], m_logits[language_id][target_id]):
                        #utt did not exist. Lets create it
                        if not utt_id in S:
                            S[utt_id] = [s]

                        #utt exists. Lets concatenate
                        #elif(config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT] == 0):
                        else:
                            #S[utt_id] += [s]
                            pass

                S, _ = self.__shrink_and_average(S)

                m_logits[language_id][target_id] = []
                new_batch_id[language_id][target_id] = []

                #iterate over all uttid again
                for idx, (utt_id, _) in enumerate(S.items()):
                    m_logits[language_id][target_id] += [S[utt_id]]
                    new_batch_id[language_id][target_id].append(utt_id)

        return new_batch_id

    def __shrink_and_average(self, S, L=None):

        avg_S={}; avg_P={}; avg_L={}; avg_O={}

        for utt_id, _ in S.items():

            #computing minimum L
            min_length = sys.maxsize
            #sys.maxint
            for utt_prob in S[utt_id]:
                if(utt_prob.shape[0] < min_length):
                    min_length = utt_prob.shape[0]

            for idx, (utt_prob) in enumerate(S[utt_id]):
                if(utt_id not in avg_S):

                    avg_S[utt_id] = S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))

                    if(L):
                        avg_L[utt_id] = L[utt_id][0:min_length][:]/float(len(L[utt_id]))
                else:
                    avg_S[utt_id] += S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))

                    if(L):
                        avg_L[utt_id] += L[utt_id][0:min_length][:]/float(len(L[utt_id]))
        return avg_S, avg_L

    def __store_results(self, config, uttids, logits, epoch):

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if(len(config[constants.CONF_TAGS.LANGUAGE_SCHEME]) > 1):
                results_dir = os.path.join(config[constants.CONF_TAGS.MODEL_DIR], language_id)
            else:
                results_dir = config[constants.CONF_TAGS.MODEL_DIR]
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            for target_id, _ in target_scheme.items():
                    writeScp(os.path.join(results_dir, "epoch%02d_cv_logit_%s.scp" % (epoch, target_id)), 
                             uttids[language_id][target_id],
                             writeArk(os.path.join(results_dir, "epoch%02d_cv_logit_%s.ark" % (epoch,target_id)), 
                                      logits[language_id][target_id], 
                                      uttids[language_id][target_id]))


    
    def __restore_weights(self):

        alpha = 1
        best_avg_ters = float('Inf')

        if self.__config[constants.CONF_TAGS.CONTINUE_CKPT]:

            print(80 * "-")
            print("restoring weights....")
            print(80 * "-")
            #restoring all variables that should be loaded during adaptation stage (all of them except adaptation layer)
            if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                    != constants.SAT_TYPE.UNADAPTED and \
                    (self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                            == constants.SAT_SATGES.TRAIN_SAT or self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                            == constants.SAT_SATGES.TRAIN_DIRECT) and not \
                    self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.CONTINUE_CKPT_SAT]:

                print("partial restoring....")
                vars_to_load=[]
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    if (not constants.SCOPES.SPEAKER_ADAPTAION in var.name):
                        vars_to_load.append(var)

                print("var list:")
                for var in vars_to_load:
                    print(var.name)

                saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH], var_list=vars_to_load)
                saver.restore(self.__sess, self.__config[constants.CONF_TAGS.CONTINUE_CKPT])

                #lets track all the variables again...
                alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config[constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])

                alpha += 1

                self.__lrscheduler.set_epoch(alpha)
            else:

                vars_to_load=[]
                if(self.__config[constants.CONF_TAGS.DIFF_NUM_TARGET_CKPT]):
                    print("partial restoring....")
                    print("var list:")
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        if(constants.SCOPES.OUTPUT not in var.name):
                            vars_to_load.append(var)
                else:
                    print("total restoring....")
                    print("var list:")
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        vars_to_load.append(var)

                for var in vars_to_load:
                    print(var)

                saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH], var_list=vars_to_load)
                saver.restore(self.__sess, self.__config[constants.CONF_TAGS.CONTINUE_CKPT])

                if True:
                    self.__lrscheduler.resume_from_log()
                else:
                    alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config[constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])

                    num_val = 0
                    acum_val = 0

                    with open(self.__config[constants.CONF_TAGS.CONTINUE_CKPT].replace(".ckpt",".log")) as input_file:
                        for line in input_file:
                            if (constants.LOG_TAGS.VALIDATE in line):
                                acum_val += float(line.split()[4].replace("%,",""))
                                num_val += 1

                        self.__ter_buffer[len(self.__ter_buffer) - 1]= acum_val / num_val

                        best_avg_ters=acum_val / num_val

                        if(alpha > 1):

                            new_log=self.__config[constants.CONF_TAGS.CONTINUE_CKPT][:-7]+"%02d" % (alpha-1,)+".log"

                            with open(new_log) as input_file:
                                for line in input_file:
                                    if (constants.LOG_TAGS.VALIDATE in line):
                                        acum_val += float(line.split()[4].replace("%,",""))
                                        num_val += 1

                            self.__ter_buffer[0]= acum_val / num_val

                alpha += 1

        print(80 * "-")

        #we want to store everything
        saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH])


        return saver, alpha, best_avg_ters

    def __generate_logs(self, cv_stats, tr_stats, epoch, lr_rate, tic):

        self.__info("Epoch %d finished in %.0f minutes" % (epoch, (time.time() - tic)/60.0))

        with open("%s/epoch%02d.log" % (self.__config["model_dir"], epoch), 'w') as fp:
            fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic)/60.0, lr_rate))

            for language_id, target_scheme in cv_stats.items():
                if len(cv_stats) > 1:
                    print("Language: "+language_id)
                    fp.write("Language: "+language_id)

                for target_id, stats in target_scheme.items():
                    if target_id == 'n':
                        continue

                    if len(target_scheme) > 2:
                        print("\tTarget: %s" % (target_id))
                        fp.write("\tTarget: %s" % (target_id))

                    tr_string = "\t\tTrain    "
                    cv_string = "\t\tValidate "
                    for key in tr_stats[language_id][target_id]:
                        tr_stat = tr_stats[language_id][target_id][key]
                        cv_stat = cv_stats[language_id][target_id][key]
                        if key in ['ter', 'acc']:
                            tr_stat *= 100.
                            cv_stat *= 100.
                        tr_string += key + ": {:.1f} ".format(float(tr_stat))
                        cv_string += key + ": {:.1f} ".format(float(cv_stat))

                    print(tr_string)
                    print(cv_string)
                    fp.write(tr_string)
                    fp.write(cv_string)
                    fp.write(self.__lrscheduler.get_status())

        # get status from LRScheduler
        self.__info(self.__lrscheduler.get_status())


    def __print_counts_debug(self, epoch, batch_counter, total_number_batches, batch, data_queue):

        args = (epoch, batch_counter, total_number_batches, batch['size'], batch['cost'])
        print("epoch={} batch={}/{} size={} batch_cost={}".format(*args))
        print("batch ",batch_counter," of ",total_number_batches,"size ",batch['size'],
              "queue ",data_queue.empty(),data_queue.full(),data_queue.qsize())

        print("ters: ")
        print(batch_ters)

    def __prepare_feed(self, data, lr_rate = None):

        x_batch, y_batch, z_batch, sat_batch = data

        if self.__config[constants.CONF_TAGS.DEBUG] == True:
            print("")
            print("the following batch_id is prepared to be processed...")
            print(x_batch[1])
            print("size batch x:")
            for element in x_batch[0]:
                print(element.shape)

            print("sizes batch y:")
            for language_id, target in y_batch[0].items():
                for target_id, content in target.items():
                    print(content[2])

            print("")

        # it contains the actual value of x
        batch_id = x_batch[1]
        x_batch  = x_batch[0]

        batch_size = len(x_batch)

        current_lan_index = 0
        for language_id, language_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if (language_id == y_batch[1]):
                index_correct_lan = current_lan_index
            current_lan_index += 1

        y_batch_list = []
        for _, value in y_batch[0].items():
            for _, value in value.items():
                y_batch_list.append(value)

        if len(y_batch_list) < self.max_targets_layers:
           for count in range(self.max_targets_layers- len(y_batch_list)):
               y_batch_list.append(y_batch_list[0])

        #eventhough self.__model.labels will be equal or grater we will use only until_batch_list
        feed = {i: y for i, y in zip(self.__trainer.labels, y_batch_list)}

        #TODO remove this prelimenary approaches
        feed[self.__model.feats] = x_batch
        #TODO fix this and allow passing parameter?
        #feed[self.__model.temperature] = 1.0 # float(config[constants.CONF_TAGS_TEST.TEMPERATURE])
        #feed[self.__teacher_model.temperature] = 1.0

        #it is training
        if lr_rate:
            feed[self.__trainer.lr_rate] = lr_rate
            feed[self.__model.is_training_ph] = True
        else:
            feed[self.__model.is_training_ph] = False

        if sat_batch:
            feed[self.__model.sat] = sat_batch

        if self.__teacher_model is not None:
            feed[self.__teacher_model.feats] = x_batch
            feed[self.__teacher_model.is_training_ph] = self.train_teacher

            if sat_batch:
                feed[self.__teacher_model.sat] = sat_batch

        # TODO perhaps use separate data loader so we don't need all this jazz
        if z_batch:
            z_batch_list = []
            for lang in z_batch[0].values():
                for value in lang.values():
                    z_batch_list.append(value)
            #feed += {i: z[1] for i, z in zip(self.__model.verify, z_batch_list)}

            # Pick out just the first language [0] values array [1]
            feed[self.__trainer.verify] = np.array(z_batch_list[0][1], dtype=np.int32)

        return feed, batch_size, index_correct_lan, batch_id

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)

    def __mat2list(self, a, seq_len):
        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]
