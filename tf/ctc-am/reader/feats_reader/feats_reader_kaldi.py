import sys
import constants
import numpy as np
from utils.fileutils import debug
from reader.feats_reader.feats_reader import FeatsReader
from utils.fileutils.kaldi import readMatrixByOffset
from utils.fileutils.kaldi import read_scp_info


class FeatsReaderKaldi(FeatsReader):

    def __init__ (self, info_set, config):

        self.__config = config

        #constructing parent class and creating self.list_files and stroing self._info_set
        super(FeatsReaderKaldi, self).__init__(info_set, self.__config[constants.CONF_TAGS.DATA_DIR],
                                               self.__config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF], "scp")

        #getting feat in dict format
        feat_dict_info_languages = self.__read_dict_info_languages()

        print("ordering all languages (from scratch) "+info_set+" batches... \n")
        self._batches_x, self._batches_id = self.__create_ordered_batches_all_languages(
            feat_dict_info_languages,
            config[constants.CONF_TAGS.LSTM_TYPE],
            config[constants.CONF_TAGS.BATCH_SIZE],
        )
        #print(self._batches_x)
        #print(self._batches_id)

    #TODO check augmentation this idea is kinda ok, but should take a closer look
    def change_source(self, new_source_positions):

        if self._info_set != 'train':
            raise ValueError("this option is only available for train set")

        # we need to recreate and refill the dictionary becuase we removed it before
        feat_dict_info_languages={}
        for language_id, scp_path in self._language_augment_scheme.items():
            print(80 * "-")
            print("preparing dictionary for "+language_id+"...\n")
            feat_dict_info_languages[language_id] = read_scp_info(scp_path[new_source_positions[language_id]])

        print(80 * "-")
        self._batches_x, self._batches_id = self.__create_ordered_batches_all_languages(
            feat_dict_info_languages,
            self.__config[constants.CONF_TAGS.LSTM_TYPE],
            self.__config[constants.CONF_TAGS.BATCH_SIZE],
        )

    #read batch idx. Input: batch index. Output: batch read with feats
    def read(self, idx):

        number_of_utt = len(self._batches_x[idx])

        #TODO remove this asap (just sanity check)
        uttid_check=[]

        if self._augmenter.concatenate == 1:
            max_utt_len = max(x[2] for x in self._batches_x[idx])
            feat_dim = self._batches_x[idx][0][3]
            tmpx = np.zeros((number_of_utt, max_utt_len, feat_dim), np.float32)

            #TODO remove uttid asap(just sanitychek)
            for i, x in enumerate(self._batches_x[idx]):
                feat, uttid = self.read_one(x)
                tmpx[i, :len(feat), :] = feat
                uttid_check.append(uttid)
        else:
            max_utt_len = max(sum(int(utt[2]) for utt in x) for x in self._batches_x[idx])
            feat_dim = self._batches_x[idx][0][0][3]
            tmpx = np.zeros((number_of_utt, max_utt_len, feat_dim), np.float32)
            
            for i, x in enumerate(self._batches_x[idx]):
                feats = []
                for utt in x:
                    feat, uttid = self.read_one(utt)
                    feats.append(feat)
                    uttid_check.append(uttid)
                feat = np.concatenate(feats, axis = 0)
                tmpx[i, :len(feat), :] = feat

        return tmpx, uttid_check

    def read_one(self, x):

        arkfile, offset, feat_len, feat_dim, augment, uttid = x

        feat = readMatrixByOffset(arkfile, offset)
        feat = self._augmenter.augment(feat, augment)

        # sanity check that the augmentation is ok
        if feat_len != feat.shape[0] or feat_dim != feat.shape[1]:
            raise ValueError("invalid shape")

        return feat, uttid


    def __create_ordered_batches_all_languages(self, feat_dict_info_languages, lstm_type, batch_size):

        all_zipped_batches = []

        #coloring every batch with its language
        for language, feat_dict_info in feat_dict_info_languages.items():
            batch_x_language, batch_id_language = self.__create_ordered_batches(feat_dict_info, lstm_type, batch_size)
            batch_id_language_c = list(zip(batch_id_language, [language]*len(batch_id_language)))
            all_zipped_batches = all_zipped_batches + list(zip(batch_x_language, batch_id_language_c))

        #unzip
        batch_x, batch_id = list(zip(*all_zipped_batches))

        return batch_x, batch_id

    def __read_dict_info_languages(self):

        feat_dict_info_languages = {}

        for language, scp_path in self._language_augment_scheme.items():
            print("preparing dictionary for "+language+"...\n")
            feat_dict_info_languages[language] = read_scp_info(scp_path[0])
            if len(feat_dict_info_languages[language]) == 0:
                raise ValueError("feature file ({}) for lanugage: {} is void".format(scp_path[0], language))

        return feat_dict_info_languages

    # it creates batches and returns a template of batch_ids that will be used externally
    # to create other readers (or maybe something else)
    def __create_ordered_batches(self, feat_info, lstm_type, batch_size):

        # Add augmentation info to feat info
        feat_info = self._augmenter.preprocess(feat_info)

        # sort the list by length
        feat_info = sorted(feat_info, key = lambda x: (x[3], x[0]))
        
        # Random batches takes 108 minutes rather than 8, with lower accuracy...
        #from random import shuffle
        #shuffle(feat_info)

        # CudnnLSTM requires batches of even sizes
        batches_x, batches_id = self.__make_batches_info(feat_info, batch_size, even = lstm_type == "cudnn")

        return batches_x, batches_id

    #It creates an even number of batches: recieves all the data and batch size
    def __make_batches_info(self, feat_info, batch_size, even = True):
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """

        # TODO: Make potentially un-even batches work
        if not even:
            raise NotImplementedError

        batch_x, uttids = [], []
        L = len(feat_info)
        idx = 0
        width = self._augmenter.concatenate

        while idx < L:
            # find batch with even size, and with maximum size of batch_size
            j = idx + 1
            target_len = feat_info[idx][3]

            while j < min(idx + (batch_size * width), L) and feat_info[j][3] == target_len:
                j += 1

            if j - idx >= width:

                xinfo, uttid = self.__get_batch_info(feat_info, idx, j - idx, width)

                batch_x.append(xinfo)
                uttids.append(uttid)

            idx = j
        return batch_x, uttids

    #contruct one batch of data
    def __get_batch_info(self, feat_info, start, height, width = 1):
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """
        uttid = [[] for i in range(height // width)]
        xinfo = [[] for i in range(height // width)]

        for i in range(height):

            uttid_aux, arkfile, offset, feat_len, feat_dim, augment = feat_info[start + i]
            item = (arkfile, offset, feat_len, feat_dim, augment, uttid_aux)

            
            if width > 1:
                xinfo[i % (height // width)].append(item)
                uttid[i % (height // width)].append(uttid_aux)
            else:
                xinfo[i] = item
                uttid[i] = uttid_aux

        return xinfo, uttid

