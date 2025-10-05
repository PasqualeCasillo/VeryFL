# model/ModelFactory.py
from .resnet import *
from .VGG import *
from .alexnet import createAlexNet
from .simple_cnn import get_simple_cnn
from .SignAlexNet import get_sign_alexnet
from .PowerGridMLP import get_powergrid_mlp  # <--- NUOVO
import logging 

logger = logging.getLogger(__name__)

class ModelFactory:
    '''
    Server as a factory for user to get predefined neural network.
    '''
    def __init__(self) -> None:
        return
    
    def get_model(self, model, class_num, args={}):
        if model == 'resnet18':
            return ResNet18(class_num)
        elif model == 'resnet34':
            return ResNet34(class_num)
        elif model == 'resnet50':
            return ResNet50(class_num)
        elif model == 'resnet101':
            return ResNet101(class_num)
        elif model == 'resnet152':
            return ResNet152(class_num)
        elif model == 'alexnet':
            return createAlexNet(class_num)
        elif model == 'simpleCNN':
            return get_simple_cnn(class_num)
        elif model == 'VGG_A':
            return VGG_A(class_num=class_num)
        elif model == 'VGG_B':
            return VGG_B(class_num=class_num)
        elif model == 'VGG_D':
            return VGG_D(class_num=class_num)
        elif model == 'VGG_E':
            return VGG_E(class_num=class_num)
        elif model == 'PowerGridMLP':  # <--- NUOVO
            input_dim = args.get('input_dim', 128)
            return get_powergrid_mlp(class_num=class_num, input_dim=input_dim)
        else:
            logger.warn("ModelFactory received an unknown model %s", model)
            raise Exception(f"Unrecognized Model: {model}")
    
    def get_sign_model(self, model, class_num, in_channels, watermark_args):
        if model == 'SignAlexNet':
            return get_sign_alexnet(class_num, in_channels, watermark_args)
        else:
            logger.error("ModelFactory received an unknown model %s", model)
            raise Exception(f"Unrecognized Model")