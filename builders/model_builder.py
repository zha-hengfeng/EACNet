import importlib

# some SOTA methods
from model.resnet18 import resnet18
from model.ESPNet import ESPNet
from model.ESPNetv2 import EESPNet_Seg
from model.CGNet import Context_Guided_Network
from model.RPNet import RPNet
# init try , some maybe failure
from model.EACNet_other import EACNet
# EACNet build base DABNet
from model.DABNet import DABNet
from model.EACNet_nl_dab import EACNet_DABv1
from model.EACNet_DAB import EACNet_DABv2
# EACNet build base ERFNet
from model.ERFNet import Net
from model.ERFNet_enc import ERFNet_ENC
from model.EACNet_nl_erf import EACNet_ERFv1
from model.EACNet_ERF import EACNet_ERFv2
# EACNet build base ENet
from model.ENet import ENet, ENet_ENC, EACNet_ENet
#EACNet build base resnet
from model.EACNet_ResNet_18 import EACNet_ResNet_18


def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        print("bulid DABNet!!!")
        return DABNet(classes=num_classes)

    if model_name == 'EACNet_nl_dab':
        print("bulid EACNet_DABv1!!!")
        return EACNet_DABv1(classes=num_classes)

    if model_name == 'EACNet_DABv2':
        print("bulid EACNet_DABv2!!!")
        return EACNet_DABv2(classes=num_classes)

    if model_name == 'ERFNet':
        print("bulid ERFNet!!!")
        return Net(num_classes=num_classes)

    if model_name == 'ERFNet_ENC':
        print("bulid ERFNet!!!")
        return ERFNet_ENC(num_classes=num_classes)

    if model_name == 'EACNet_ERFv1':
        print("bulid EACNetv1(ERFNet)!!!")
        return EACNet_ERFv1(num_classes=num_classes)

    if model_name == 'EACNet_ERFv2':
        print("bulid EACNetv2(ERFNet)!!!")
        return EACNet_ERFv2(num_classes=num_classes)

    if model_name == 'resnet18':
        print("bulid resnet18!!!")
        return resnet18()

    if model_name == 'ESPNet':
        print("bulid ESPNet!!!")
        return ESPNet()

    if model_name == 'ESPNetv2':
        print("bulid ESPNetv2!!!")
        return EESPNet_Seg()

    if model_name == 'ENet':
        print("bulid ENet!!!")
        return ENet()

    if model_name == 'CGNet':
        print("bulid CGNet!!!")
        return Context_Guided_Network()

    if model_name == 'RPNet':
        print("bulid RPNet!!!")
        return RPNet(20)

    if model_name == 'ENet_ENC':
        print('build ENet_ENC')
        return ENet_ENC(n_classes=num_classes)

    if model_name == 'EACNet_ENet':
        print('build EACNet_ENet')
        return EACNet_ENet(n_classes=num_classes)

    if model_name == 'EACNet_ResNet-18-ENC':
        print("=====> Build EACNet_ResNet-18-ENC !")
        return EACNet_ResNet_18(num_classes=num_classes, encoder_only=True)

    if model_name == 'EACNet_ResNet-18':
        print("=====> Build EACNet_ResNet-18 !")
        return EACNet_ResNet_18(num_classes=num_classes, encoder_only=False)