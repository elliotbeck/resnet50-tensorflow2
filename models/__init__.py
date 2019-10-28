from models.densenet import DenseNet
from models.resnet import ResNet50

def get_model(name, config):
    if name == "densenet121":
        return DenseNet(config.num_classes, config.densenet_weights, config)
    elif name == "ResNet50":
        return ResNet50(config.num_classes, config.resnet_weights, config)
    else:
        raise NotImplementedError