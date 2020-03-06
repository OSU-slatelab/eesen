from models_16.deep_bilstm import DeepBidirRNN
from models_16.resnet_bilstm import ResnetBilstm


import constants

#it returns an object lm_reader that internaly will manage all the data
#client will be agnostic for the internals
def create_model(config):

    models = {
        constants.MODEL_NAME.DEEP_BILSTM: DeepBidirRNN,
        constants.MODEL_NAME.RESNET_BILSTM: ResnetBilstm,
    }

    try:
        from models_16.resnet import Resnet
        models[constants.MODEL_NAME.RESNET] = Resnet
    except:
        pass

    if config[constants.CONF_TAGS.MODEL] in models:
        return models[config[constants.CONF_TAGS.MODEL]](config)
    else:
        raise ValueError('Model must be one of ' + str(models.keys()))

