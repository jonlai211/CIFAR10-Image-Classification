import torchvision.models as models
from pytorch_cifar_classification.models.origin_net import Net as OriginNet
from pytorch_cifar_classification.models.modified_net import Net as ModifiedNet
from pytorch_cifar_classification.utils.model_utils import print_model_info, print_model_memory_info
import torch

origiNet = OriginNet()
modifiedNet = ModifiedNet()
ResNet18 = models.resnet18(pretrained=True)
DenseNet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
VGG16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
InceptionV3 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
GoogLeNet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
MobileNetV2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
MobileNetV2.classifier[1] = torch.nn.Linear(MobileNetV2.last_channel, 10)

input_size = [1, 3, 32, 32]

origiNet_path = 'models/originNet_2023-12-07_15-54-16.pt'
modifiedNet_path = 'models/'
ResNet18_path = 'models/ResNet18_2023-12-07_20-51-09.pt'

print("OriginNet:")
print_model_info(origiNet, origiNet_path)

print("ModifiedNet:")
print_model_info(modifiedNet, modifiedNet_path)

print("ResNet18:")
print_model_info(ResNet18, ResNet18_path)

print("DenseNet121:")
print_model_info(DenseNet121, '')
print_model_memory_info(DenseNet121, input_size)

print("VGG16:")
print_model_info(VGG16, '')
print_model_memory_info(VGG16, input_size)

print("InceptionV3:")
print_model_info(InceptionV3, '')
print_model_memory_info(InceptionV3, input_size)

print("GoogLeNet:")
print_model_info(GoogLeNet, '')
print_model_memory_info(GoogLeNet, input_size)

print("MobileNetV2:")
print_model_info(MobileNetV2, '')
print_model_memory_info(MobileNetV2, input_size)