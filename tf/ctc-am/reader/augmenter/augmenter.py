import numpy as np
import sys, random
import random
import sys
import numpy as np
import constants
from utils.fileutils.debug import get_debug_info

def update(tup, shift, window, factor):
    tup = list(tup)
    tup[3] = int(tup[3]) // factor #(int(tup[3]) + factor - 1 - shift) // factor
    tup[4] *= window
    tup.append((shift, factor, window))
    return tuple(tup)

class Augmenter(object):
    def __init__(self, online_augment_config):
        self.factor = online_augment_config[constants.AUGMENTATION.SUBSAMPLING]
        self.window = online_augment_config[constants.AUGMENTATION.WINDOW]
        self.concatenate = online_augment_config[constants.AUGMENTATION.CONCATENATE]

    def preprocess(self, feat_info):

        #TODO @florian here you can have more room to play with augmentation option
        print("Augmenting data factor: {} and window: {}".format(self.factor, self.window))
        print(feat_info[0])

        feat_info = [update(tup, shift, self.window, self.factor)
            for shift in range(self.factor) for tup in feat_info]

        print(feat_info[0])

        return feat_info

    def augment(self, feat, augment):

        shift = augment[0]
        stride = augment[1]
        win = augment[2]

        augmented_feats = []
        for i in range(-(win // 2), win - win // 2):
            augmented_feats.append(np.roll(feat, i, axis = 0))


        # Reduce length to multiple of stride, then stride it!
        augmented_feats = np.concatenate(augmented_feats, axis = 1)
        feats_max = len(augmented_feats) - len(augmented_feats) % stride
        augmented_feats = augmented_feats[shift:feats_max:stride,]

        #roll is deprecated
        #if self.online_augment_config["roll"]:
        #    augmented_feats = np.roll(augmented_feats, random.randrange(-2,2,1), axis = 0)

        return augmented_feats

