from keras import models
from keras_contrib.layers import CRF

def create_custom_objects():
    instanceHolder = {"instance": None}
    
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
            
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    
    return {
        "ClassWrapper": ClassWrapper,
        "CRF": ClassWrapper,
        "loss": loss,
        "accuracy": accuracy}

def load_keras_model(path):
    """Utility function for saving CRF layer alongside the model.
    from https://github.com/keras-team/keras-contrib/issues/129

    :param path: Path to the Keras hdf5 file
    :returns the Keras part of Minuet
    """
    return models.load_model(path, create_custom_objects())
