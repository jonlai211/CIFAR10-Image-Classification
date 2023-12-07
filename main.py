import torchvision.models as models
from pytorch_cifar_classification.models.origin_net import Net
from pytorch_cifar_classification.utils.model_utils import print_model_info
import torch

origiNet = Net()
ResNet18 = models.resnet18(pretrained=True)
MobileNetV2 = models.mobilenet_v2(pretrained=True)
MobileNetV2.classifier[1] = torch.nn.Linear(MobileNetV2.last_channel, 10)

origiNet_path = 'models/originNet_2023-12-07_15-54-16.pt'
ResNet18_path = 'models/ResNet18_2023-12-07_20-51-09.pt'
MobileNetV2_path = 'models/MobileNetV2.pt'

print("OriginNet:")
print_model_info(origiNet, origiNet_path)

print("ResNet18:")
print_model_info(ResNet18, ResNet18_path)

print("MobileNetV2:")
print_model_info(MobileNetV2, MobileNetV2_path)
