import os
import constants
import sys
import shutil
try:
    import h5py
except:
    pass

import time
from itertools import islice
#from multiprocessing import Process, Queue
from models.model_factory import create_model
from trainer import Trainer

#from reader.reader_queue import run_reader_queue
import numpy as np
import tensorflow as tf

from utils.fileutils.kaldi import writeArk, writeScp


class Test():

    def test_impl(self, data, config):

        self.__model = create_model(config, 'student')
        self.__trainer = Trainer(config, self.__model)

        with tf.Session() as sess:

            student_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student')
            #saver = tf.train.Saver({var.op.name[8:]: var for var in student_vars})
            saver = tf.train.Saver()

            print("restoring weights from path:")
            print(config[constants.CONFIG_TAGS_TEST.TRAINED_WEIGHTS])

            saver.restore(sess, config[constants.CONFIG_TAGS_TEST.TRAINED_WEIGHTS])

            print("weights restored")
            print(80 * "-")
            print(80 * "-")
            print("start decoding...")
            soft_probs, log_soft_probs, log_likes, logits, batches_id = self.__create_result_containers(config)

            ntest, test_costs, test_ters, test_acc, ntest_labels = self.__create_counter_containers(config)


            #process, data_queue, test_x = self.__generate_queue(config, data)
            #process.start()

            start_time = time.time()

            #batch_counter = 0
            total_number_batches = len(data['x'].get_batches_id())
            #batches_id = np.array(data['x'].get_batches_id()).flatten()
            samples = 0
            total_len = 0
            ones_len = 0
            nonones_len = 0
            batches_id = []

            request_list = self.__prepare_request_list(config)
            assert next(iter(request_list)) == 'seq_len'

            #while True:
            for i in range(total_number_batches):

                #data = data_queue.get()
                #if data is None:
                #    break

                batch_data = {'x': data['x'].read(i)}
                if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
                    batch_data['y'] = data['y'].read(i)
                if config[constants.CONF_TAGS.COMPUTE_ACC]:
                    batch_data['alt'] = [a.read(i) for a in data['alt']]
                if config[constants.CONFIG_TAGS_TEST.VERIFY_FILE]:
                    batch_data['z'] = data['z'].read(i)

                feed, batch_size, index_correct_lan, batch_id = self.__prepare_feed(batch_data, config)
                batches_id.extend(batch_id)
                samples += batch_size

                outputs = sess.run([request['var'] for request in request_list.values()], feed)
                for i, name in enumerate(request_list):
                    if request_list[name]['type'] == 'batch_stat':
                        request_list[name]['result'] = outputs[i]
                    elif request_list[name]['type'] == 'counter':
                        request_list[name]['result'] += outputs[i][0][0]
                    elif request_list[name]['type'] == 'container':
                        seq_len = request_list['seq_len']['result']
                        for uttid, mat in zip(batch_id, self.__mat2list(outputs[i][0][0], seq_len)):
                            if uttid in request_list[name]['result']:
                                length = len(request_list[name]['result'][uttid])
                                request_list[name]['result'][uttid] += mat[:length]
                            else:
                                request_list[name]['result'][uttid] = mat
                    else:
                        raise ValueError("Type must be one of 'batch_stat', 'container', 'counter'")

                    #if request_list[i]['name'] == 'verify':
                    #    print(batch_size)
                    #    print(item[0][0])
                    #    sys.exit()

                idxs, vals, dense_shape = request_list['labels']['result']
                #print(vals)

                if config[constants.CONFIG_TAGS_TEST.VERIFY_FILE]:
                    zvals = feed[self.__trainer.verify]
                    #for i, item in enumerate(batch_data['z']):
                    #    if item:
                    for idx in idxs:
                        if zvals[idx[0]] == 0:
                            ones_len += 1
                        else:
                            nonones_len += 1

                # Since the labels are sparse, length of values is total number of terms
                if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
                    #total_len += self.__get_label_len(request_list['labels']['result'])
                    total_len += len(idxs)

                #batch_soft_probs, batch_log_soft_probs, batch_seq_len, batch_logits = outputs[0:4]

                #TODO clean this up
                #batch_ters = None
                #batch_log_likes = None
                #batch_acc = None

                #if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                #    batch_ters, batch_cost, batch_log_likes = outputs[4:7]

                #elif config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and not config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                #    batch_ters, batch_cost = outputs[4:6]

                #elif not config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                #    batch_log_likes = outputs[4]

                #if config[constants.CONF_TAGS.COMPUTE_ACC]:
                #    batch_acc = outputs[-1]

                #if not config[constants.CONFIG_TAGS_TEST.ONLINE_STORAGE]:

                #    self.__update_probs_containers(config, batch_id, batches_id, request_list, soft_probs,
                #        log_soft_probs, logits, log_likes)

                #else:
                #    self.__store_online(config, batch_id, request_list)

                #if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
                #    self.__update_counters(config, batch_size, ntest, y_batch, ntest_labels, request_list,
                #        test_ters, test_acc, test_costs)

                #if batch_counter % 100 == 0 and batch_counter > 0:
                #    complete= (float(batch_counter)/float(total_number_batches))*100
                #    print("Test done: %.1f (batch %d of %d)" % (complete, batch_counter, total_number_batches))

                #batch_counter += 1

            #process.join()
            #process.terminate()
            print("done decoding")
            print(80 * "-")
            print(80 * "-")

            samp = config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF][constants.AUGMENTATION.SUBSAMPLING]

            for name in request_list:
                x = request_list[name]['result']
                if name == 'ters':
                    request_list[name]['result'] = x * 100.0 / float(total_len)
                elif name == 'acc':
                    request_list[name]['result'] = x * 100.0 / float(samples)
                elif name == 'ones_ter':
                    request_list[name]['result'] = x * 100.0 / float(ones_len)
                elif name == 'nonones_ter':
                    request_list[name]['result'] = x * 100.0 / float(nonones_len)
                elif name == 'verify_acc':
                    TP, TN, FP, FN = request_list[name]['result'].tolist()
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    request_list[name]['result'] = 200. * precision * recall / (precision + recall)
                    #up = TP * TN - FP * FN
                    #dn = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    #request_list[name]['result'] = 100*up/dn
                elif request_list[name]['type'] == 'counter':
                    request_list[name]['result'] /= float(samples)
                elif request_list[name]['type'] == 'container':
                    if samp > 0:
                        for uttid in request_list[name]['result']:
                            request_list[name]['result'][uttid] /= samp

            request_list['precision'] = {'type': 'counter', 'result': precision * 100}
            request_list['recall'] = {'type': 'counter', 'result': recall * 100}
            

            #if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:

            #    for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            #        for target_id, _ in target_scheme.items():
            #            test_costs[language_id][target_id] = test_costs[language_id][target_id] / float(ntest[language_id])

            #    for language_id, target_scheme in test_ters.items():
            #        for target_id, cv_ter in target_scheme.items():
            #            test_ters[language_id][target_id] = cv_ter/float(ntest_labels[language_id][target_id])
            #            test_acc[language_id][target_id] = test_acc[language_id][target_id]/float(ntest[language_id])


            self.__print_logs(config, request_list, start_time)

            if not config[constants.CONFIG_TAGS_TEST.ONLINE_STORAGE]:
                #if config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF][constants.AUGMENTATION.SUBSAMPLING] > 0:
                #    print(len(batches_id))
                #    batches_id = self.__average_over_augmented_data(config, batches_id, 
                #       soft_probs, log_soft_probs, log_likes, logits)

                self.__store_results(config, batches_id, request_list)
                    #soft_probs, log_soft_probs, log_likes, logits)


    def __prepare_request_list(self, config):

        request_list = {
            'seq_len': {'type': 'batch_stat', 'var': self.__model.seq_len},
            'soft_prob': {'type': 'container', 'var': self.__trainer.softmax_probs},
            'log_soft': {'type': 'container', 'var': self.__trainer.log_softmax_probs},
            'logits': {'type': 'container', 'var': self.__trainer.logits},
            'framecount': {'type': 'counter', 'var': self.__trainer.framecount},
            'peakcount': {'type': 'counter', 'var': self.__trainer.peakcount},
            'tdloss': {'type': 'counter', 'var': self.__trainer.tdloss},
        }

        if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
            request_list['labels'] = {'type': 'batch_stat', 'var': self.__trainer.labels[0]}
            request_list['ters'] = {'type': 'counter', 'var': self.__trainer.ters}
            request_list['costs'] = {'type': 'counter', 'var': self.__trainer.debug_costs}
            request_list['all_ters'] = {'type': 'batch_stat', 'var': self.__trainer.all_ters[0]}

        if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
            request_list['likelihoods'] = {'type': 'container', 'var': self.__trainer.log_likelihoods}

        if config[constants.CONF_TAGS.COMPUTE_ACC]:
            request_list['acc'] = {'type': 'counter', 'var': self.__trainer.acc}

        if config[constants.CONFIG_TAGS_TEST.VERIFY_FILE]:
            request_list['verify_acc'] = {'type': 'counter', 'var': self.__trainer.verify_acc}

            if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
                request_list['ones_ter'] = {'type': 'counter', 'var': self.__trainer.ones_ter}
                #request_list['ones_count'] = {'type': 'counter', 'var': self.__trainer.ones_count}
                request_list['nonones_ter'] = {'type': 'counter', 'var': self.__trainer.nonones_ter}
                #request_list['nonones_count'] = {'type': 'counter', 'var': self.__trainer.nonones_count}

        # Initialize results
        for name in request_list:
            if name == 'verify_acc':
                request_list[name]['result'] = np.array([0., 0., 0., 0.])
            elif request_list[name]['type'] == 'counter':
                request_list[name]['result'] = 0
            elif request_list[name]['type'] == 'container':
                request_list[name]['result'] = {}

        return request_list

    def  __store_online(self, config, batch_id, batch_soft_probs,
                        batch_log_soft_probs, batch_logits, batch_log_likes):

        language_idx = 0
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():

            target_idx=0
            for target_id, num_targets in target_scheme.items():


                if len(config[constants.CONF_TAGS.LANGUAGE_SCHEME].keys()) > 1:
                    root_store_path = os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR], language_id)

                    if not os.path.exists(root_store_path):
                        os.makedirs(root_store_path)
                else:
                    root_store_path = config[constants.CONFIG_TAGS_TEST.RESULTS_DIR]

                #log likes
                if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:

                    if len(target_scheme.keys()) > 1:
                        log_like_store_path = os.path.join(root_store_path, "log_like_"+target_id+".hdf5")
                    else:
                        log_like_store_path = os.path.join(root_store_path, "log_like.hdf5")

                    self.__partial_store_hdf5(log_like_store_path, batch_log_likes[language_idx][target_idx],
                                              config[constants.CONFIG_TAGS_TEST.COUNT_AUGMENT], batch_id, config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT])

                #soft probs
                if len(target_scheme.keys()) > 1:
                    soft_prob_path = os.path.join(root_store_path, "soft_prob_"+target_id+".hdf5")
                else:
                    soft_prob_path = os.path.join(root_store_path, "soft_prob.hdf5")
                self.__partial_store_hdf5(soft_prob_path, batch_soft_probs[language_idx][target_idx],
                                          config[constants.CONFIG_TAGS_TEST.COUNT_AUGMENT], batch_id, config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT])

                #log_soft probs
                if len(target_scheme.keys()) > 1:
                    log_soft_prob_path = os.path.join(root_store_path, "log_soft_prob_"+target_id+".hdf5")
                else:
                    log_soft_prob_path = os.path.join(root_store_path, "log_soft_prob.hdf5")
                self.__partial_store_hdf5(log_soft_prob_path, batch_log_soft_probs[language_idx][target_idx],
                                          config[constants.CONFIG_TAGS_TEST.COUNT_AUGMENT], batch_id, config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT])

                #logits
                if len(target_scheme.keys()) > 1:
                    logit_path = os.path.join(root_store_path, "logits_"+target_id+".hdf5")
                else:
                    logit_path = os.path.join(root_store_path, "logits.hdf5")
                self.__partial_store_hdf5(logit_path, batch_logits[language_idx][target_idx], config[constants.CONFIG_TAGS_TEST.COUNT_AUGMENT],
                                          batch_id, config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT])

                target_idx += 1

            language_idx += 1

    def __partial_store_hdf5(self, path, data, batch_id_counts, batch_id, subsampled_utt):


        with h5py.File(path) as h5file:


            for idx, utt in enumerate(data):

                rolled_utt = np.roll(utt, 1, axis=1)

                #TODO would be cool to decide which one to choose
                #TODO would be cool to have a sanity check also
                if subsampled_utt > 0 :
                    if batch_id[idx] not in h5file:
                        h5file[batch_id[idx]] = rolled_utt
                else:
                    if batch_id[idx] in h5file:
                        #new_data = [np.roll(data[0][i, : batch_seq_len[i] :], 1, axis = 1) for i in range(len(data))]

                        #get stored sample
                        stored_sample = h5file[batch_id[idx]]
                        #check minimum lengthtt
                        if stored_sample.shape[0] < rolled_utt.shape[0]:
                            final_length = stored_sample.shape[0]
                        else:
                            final_length = rolled_utt.shape[0]

                        del h5file[batch_id[idx]]

                        h5file[batch_id[idx]] = stored_sample[0:final_length][:] + \
                                                rolled_utt[0:final_length][:]/float(batch_id_counts[batch_id[idx]])

                    else:
                        h5file[batch_id[idx]] = rolled_utt/float(batch_id_counts[batch_id[idx]])

    def __average_over_augmented_data(self, config, m_batches_id, request_list):
            #m_soft_probs, m_log_soft_probs, m_log_likes, m_logits):
        #new batch structure
        new_batch_id = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            new_batch_id[language_id] = {}

            for target_id, num_targets in target_scheme.items():
                if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:

                    S={}; P={}; L={}; O={};
                    #iterate over all utterances of a concrete language
                    for utt_id, s, p, l, o in zip(m_batches_id[language_id], m_soft_probs[language_id][target_id], m_log_soft_probs[language_id][target_id],
                                                  m_log_likes[language_id][target_id], m_logits[language_id][target_id]):
                        #utt did not exist. Lets create it
                        if not utt_id in S:
                            S[utt_id] = [s]; P[utt_id] = [p]; L[utt_id] = [l]; O[utt_id] = [o]

                        #utt exists. Lets concatenate
                        elif config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT] == 0:
                            S[utt_id] += [s]; P[utt_id] += [p]; L[utt_id] += [l]; O[utt_id] += [o]

                    S, P, O, L = self.__shrink_and_average(S, P, O, L)

                else:

                    S={}; P={}; O={}
                    #iterate over all utterances of a concrete language
                    for utt_id, s, p, o in zip(m_batches_id[language_id], m_soft_probs[language_id][target_id], m_log_soft_probs[language_id][target_id],
                                               m_logits[language_id][target_id]):
                        #utt did not exist. Lets create it
                        if not utt_id in S:
                            S[utt_id] = [s]; P[utt_id] = [p]; O[utt_id] = [o]

                        #utt exists. Lets concatenate
                        elif config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT] == 0:
                            S[utt_id] += [s]; P[utt_id] += [p]; O[utt_id] += [o]

                    S, P, O, _ = self.__shrink_and_average(S, P, O)

                m_soft_probs[language_id][target_id] = []
                m_log_soft_probs[language_id][target_id] = []
                m_logits[language_id][target_id] = []
                new_batch_id[language_id][target_id] = []

                if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                    m_log_likes[language_id][target_id] = []

                #iterate over all uttid again
                for idx, (utt_id, _) in enumerate(S.items()):
                    m_soft_probs[language_id][target_id] += [S[utt_id]]
                    m_log_soft_probs[language_id][target_id] += [P[utt_id]]
                    m_logits[language_id][target_id] += [O[utt_id]]
                    new_batch_id[language_id][target_id].append(utt_id)

                    if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                        m_log_likes[language_id][target_id] += [L[utt_id]]

        return new_batch_id

    def __shrink_and_average(self, S, P, O, L=None):

        avg_S={}; avg_P={}; avg_L={}; avg_O={}

        for utt_id, _ in S.items():

            #computing minimum L
            min_length = sys.maxsize
            #sys.maxint
            for utt_prob in S[utt_id]:
                if utt_prob.shape[0] < min_length:
                    min_length = utt_prob.shape[0]

            for idx, (utt_prob) in enumerate(S[utt_id]):
                if utt_id not in avg_S:

                    avg_S[utt_id] = S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))
                    avg_P[utt_id] = P[utt_id][idx][0:min_length][:]/float(len(P[utt_id]))
                    avg_O[utt_id] = O[utt_id][idx][0:min_length][:]/float(len(O[utt_id]))

                    if L:
                        avg_L[utt_id] = L[utt_id][0:min_length][:]/float(len(L[utt_id]))
                else:
                    avg_S[utt_id] += S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))
                    avg_P[utt_id] += P[utt_id][idx][0:min_length][:]/float(len(P[utt_id]))
                    avg_O[utt_id] += O[utt_id][idx][0:min_length][:]/float(len(O[utt_id]))

                    if L:
                        avg_L[utt_id] += L[utt_id][0:min_length][:]/float(len(L[utt_id]))
        return avg_S, avg_P, avg_O, avg_L


    def __update_probs_containers(self, config,
                                  batch_id, m_batches_id,
                                  request_list,
                                  m_soft_probs,
                                  m_log_soft_probs,
                                  m_logits,
                                  m_log_likes):

        batch_seq_len = request_list['seq_len']['results']
        batch_soft_probs = request_list['soft_probs']['results']

        language_idx=0
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            m_batches_id[language_id] += batch_id
            target_idx=0
            for target_id, num_targets in target_scheme.items():

                m_soft_probs[language_id][target_id] += self.__mat2list(batch_soft_probs[language_idx][target_idx], batch_seq_len)
                m_log_soft_probs[language_id][target_id] += self.__mat2list(batch_log_soft_probs[language_idx][target_idx], batch_seq_len)
                m_logits[language_id][target_id] += self.__mat2list(batch_logits[language_idx][target_idx], batch_seq_len)

                if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                    m_log_likes[language_id][target_id] += self.__mat2list(batch_log_likes[language_idx][target_idx], batch_seq_len)
                target_idx += 1
            language_idx += 1

    def __create_counter_containers(self, config):

        ntest = {}
        test_cost = {}
        test_ters = {}
        test_acc = {}
        ntest_labels = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            ntest[language_id] = 0
            test_cost[language_id] = {}
            test_ters[language_id] = {}
            test_acc[language_id] = {}
            ntest_labels[language_id] = {}
            for target_id, _ in target_scheme.items():
                test_ters[language_id][target_id] = 0
                test_acc[language_id][target_id] = 0
                test_cost[language_id][target_id] = 0
                ntest_labels[language_id][target_id] = 0

        return ntest, test_cost, test_ters, test_acc, ntest_labels

    def __create_result_containers(self, config):

        batches_id={}
        soft_probs = {}
        log_soft_probs = {}
        log_likes = {}
        logits = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            batches_id[language_id] = []
            soft_probs[language_id] = {}
            soft_probs[language_id] = {}
            log_soft_probs[language_id] = {}
            log_likes[language_id] = {}
            logits[language_id] = {}
            for target_id, _ in target_scheme.items():
                soft_probs[language_id][target_id] = []
                log_soft_probs[language_id][target_id] = []
                log_likes[language_id][target_id]=[]
                logits[language_id][target_id]=[]

        return soft_probs, log_soft_probs, log_likes, logits, batches_id

    def __print_logs(self, config, request_list, tic):

        print("Test time: %.0f minutes\n" % ((time.time() - tic)/60.0))
        #for language_id, target_scheme in test_ters.items():
        #    if len(test_ters) > 1:
        #        fp = open(os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR],"test.log"), "w")
        #        fp.write("Test time: %.0f minutes\n" % ((time.time() - tic)/60.0))
        #        print("Language: "+language_id)
        #        fp.write("Language: "+language_id)

        #    else:
        #        fp = open(os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR],"test.log"), "w")

        #    for target_id,  test_ter in target_scheme.items():
        #        acc = test_acc[language_id][target_id]
        #        if len(target_scheme) > 1:
        #            print("\tTarget: %s" % (target_id))
        #            fp.write("\tTarget: %s" % (target_id))
                #print("\t\t Test cost: %.1f, ter: %.1f%%, acc: %.1f%%, #example: %d" % (test_cost[language_id][target_id], 100.0*test_ter, 100.0*acc, ntest[language_id]))
                #fp.write("\t\tTest cost: %.1f, ter: %.1f%%, acc: %.1f%%, #example: %d\n" % (test_cost[language_id][target_id], 100.0*test_ter, 100.0*acc, ntest[language_id]))
        output = ""
        for name in request_list:
            if request_list[name]['type'] == 'counter':
                output += "\t\t%s: %.1f%%\n" % (name, request_list[name]['result'])

        print(output)
        with open(os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR],"test.log"), "w") as fp:
            fp.write(output)


    def __store_results(self, config, uttids, request_list):
            #soft_probs, log_soft_probs, log_likes, logits):

        results_dir = config[constants.CONFIG_TAGS_TEST.RESULTS_DIR]
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        logits = request_list['logits']['result']
        writeScp(os.path.join(results_dir, "logit.scp"), logits.keys(),
                 writeArk(os.path.join(results_dir, "logit.ark"), logits.values(), logits.keys()))

        """
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if len(config[constants.CONF_TAGS.LANGUAGE_SCHEME]) > 1:
                results_dir = os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR], language_id)
            else:
                results_dir = config[constants.CONFIG_TAGS_TEST.RESULTS_DIR]
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            for target_id, _ in target_scheme.items():

                if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
                    writeScp(os.path.join(results_dir, "log_like_"+target_id+".scp"), uttids[language_id][target_id],
                             writeArk(os.path.join(results_dir, "log_like_"+target_id+".ark"), log_likes[language_id][target_id], uttids[language_id][target_id]))

                writeScp(os.path.join(results_dir, "soft_prob_"+target_id+".scp"), uttids[language_id][target_id],
                         writeArk(os.path.join(results_dir, "soft_prob_"+target_id+".ark"), soft_probs[language_id][target_id], uttids[language_id][target_id]))

                writeScp(os.path.join(results_dir, "log_soft_prob_"+target_id+".scp"), uttids[language_id][target_id],
                         writeArk(os.path.join(results_dir, "log_soft_prob_"+target_id+".ark"), log_soft_probs[language_id][target_id], uttids[language_id][target_id]))

                writeScp(os.path.join(results_dir, "logit_"+target_id+".scp"), uttids[language_id][target_id],
                         writeArk(os.path.join(results_dir, "logit_"+target_id+".ark"), logits[language_id][target_id], uttids[language_id][target_id]))
        """


    def __update_counters(self, config, batch_size, m_acum_samples, ybatch, m_acum_labels, batch_ters, m_acum_ters, batch_acc, m_acum_acc, batch_cost, m_acum_cost):


        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for idx_lan, (language_id, target_scheme) in enumerate(config[constants.CONF_TAGS.LANGUAGE_SCHEME].items()):
            if ybatch[1] == language_id:

                for idx_tar, (target_id, _) in enumerate(target_scheme.items()):
                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    m_acum_ters[language_id][target_id] += batch_ters[idx_lan][idx_tar]
                    m_acum_acc[language_id][target_id] += batch_acc[idx_lan][idx_tar]
                    m_acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0][language_id][target_id])

                    if batch_cost[idx_lan][idx_tar] != float('Inf'):
                        m_acum_cost[language_id][target_id] += batch_cost[idx_lan][idx_tar] * batch_size

        m_acum_samples[ybatch[1]] += batch_size


    #important to note that shuf = False (to be able to reuse uttid)
    def __generate_queue (self, config, data):

        #test_x, test_y, test_sat = data

        data_queue = Queue(config[constants.CONFIG_TAGS_TEST.BATCH_SIZE])

        if 'y' in data:
            if 'sat' in data:
                #x, y, sat
                process = Process(target= run_reader_queue, args= (data_queue, data['x'], data['y'], False, False, 0, test_sat))
            else:
                #x, y
                process = Process(target = run_reader_queue, args= (data_queue, test_x, test_y, False, False, 0, None))
        else:
            if test_sat:
                #x, sat
                process = Process(target = run_reader_queue, args= (data_queue, test_x, None, False, False, 0, test_sat))
            else:
                #x
                process = Process(target = run_reader_queue, args = (data_queue, test_x, None, False, False, 0, None))

        return process, data_queue, test_x


    def __prepare_feed(self, data, config):

        #TODO solve this
        #y_batch = None
        #if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
        #        != constants.SAT_TYPE.UNADAPTED:
        #    x_batch, y_batch, sat_batch = data
        #elif config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
        #    x_batch, y_batch = data
        #else:
        #    x_batch = data

        #getting the actual x_batch that is in position 0
        batch_size = len(data['x'][0])

        current_lan_index = 0

        index_correct_lan = None
        if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:

            #we just convert from dict to a list
            y_batch_list = []
            # batch{language_1;{target_1: labels, target_2: labels},
            # language_2;{target_1: labels, target_2: labels}]
            for language_id, batch_targets in data['y'][0].items():
                for targets_id, batch_targets in batch_targets.items():
                    y_batch_list.append(batch_targets)

            for language_id, language_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
                if language_id == data['y'][1]:
                    index_correct_lan = current_lan_index
                current_lan_index += 1

            #eventhough self.__model.labels will be equal or grater we will use only until_batch_list
            feed = {i: y for i, y in zip(self.__trainer.labels, y_batch_list)}

        else:
            feed = {}

        if config[constants.CONF_TAGS.COMPUTE_ACC]:
            alt_batch_list = []
            for reader in data['alt']:
                for language_id, batch_targets in reader[0].items():
                    for targets_id, batch_targets in batch_targets.items():
                        alt_batch_list.append(batch_targets)
            for i, y in zip(self.__trainer.alt, alt_batch_list):
                feed[i] = y

        if config[constants.CONFIG_TAGS_TEST.VERIFY_FILE]:
            #verify_list = []
            for language_id, batch_targets in data['z'][0].items():
                for targets_id, batch_targets in batch_targets.items():
                    verify_list = batch_targets[1]

            feed[self.__trainer.verify] = verify_list

        #TODO remove this prelimenary approaches
        #getting the actual x_batch that is in position 0
        feed[self.__model.feats] = data['x'][0]
        #feed[self.__model.temperature] = float(config[constants.CONFIG_TAGS_TEST.TEMPERATURE])
        feed[self.__model.is_training_ph] = False

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            feed[self.__model.sat] = data['sat']


        if config[constants.CONFIG_TAGS_TEST.USE_PRIORS]:
            #TODO we need to add priors also:
            #feed_priors={i: y for i, y in zip(model.priors, config["prior"])}
            print(config[constants.CONFIG_TAGS_TEST.PRIORS_SCHEME])

        #return feed, batch_size, correct_index, uttid_batch
        return feed, batch_size, index_correct_lan, data['x'][1]#, data['y']

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)

    def __mat2list(self, a, seq_len):

        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

    def __chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
