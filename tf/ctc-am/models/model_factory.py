from models.achen import Achen
from models.achen_conv import AchenConv
from models.achen_sum import AchenSum
from models.convnet import ConvNet
from models.deep_rnn import DeepRNN
from models.dnn import DNN
#from models.verify import Verify
#from models.arcnet_video import ArcNetVideo

import tensorflow as tf
import constants

#it returns an object lm_reader that internaly will manage all the data
#client will be agnostic for the internals
def create_model(config, scope = 'student', teacher = None, train = True):

    models = {
        constants.MODEL_NAME.CONVNET     : ConvNet,
        constants.MODEL_NAME.ACHEN       : Achen,
        constants.MODEL_NAME.ACHEN_CONV  : AchenConv,
        constants.MODEL_NAME.ACHEN_SUM   : AchenSum,
        constants.MODEL_NAME.DNN         : DNN,
        #constants.MODEL_NAME.ARCNET_VIDEO: ArcNetVideo,
    }

    try:
        from models.resnet import Resnet
        models[constants.MODEL_NAME.RESNET] = Resnet
    except:
        pass


    with tf.variable_scope(scope):
        if config[constants.CONF_TAGS.MODEL] in models:
            return models[config[constants.CONF_TAGS.MODEL]](config)
        else:
            return DeepRNN(config, scope, teacher, train)

    """
    if config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_BILSTM:
        return DeepBidirRNN(config, 'bidirectional')
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_LSTM:
        return DeepBidirRNN(config, 'unidirectional')
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_BILSTM_RELU:
        return DeepBidirRNNRelu(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.RESNET:
        return Resnet(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.VERIFY:
        return Verify(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ACHEN:
        return Achen(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ACHEN_SUM:
        return AchenSum(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.CONVNET:
        return ConvNet(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ACHEN_CONV:
        return AchenConv(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ARCNET_VIDEO:
        return ArcNetVideo(config)
    else:
        raise ValueError("model selected doesn't exist")
    """
