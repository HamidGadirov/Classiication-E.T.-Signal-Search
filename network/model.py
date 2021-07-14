import torchvision.models as models
import torch
import torch.nn as nn
import sys
sys.path.append('/data/p303872/SET/code/src/network')
import efficientnet.model as effnet
def build_model(backbone):
    resnet18 = models.resnet18(pretrained=True)
    resnet34 = models.resnet34(pretrained=True)
    resnet101 = models.resnet101(pretrained=True)
    densenet121 = models.densenet121(pretrained=True)
    densenet161 = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    resnext101_32x4d = models.resnext101_32x8d(pretrained=True)
    model_dict = {"resnet18": resnet18,
                   "resnet34": resnet34,
                   "resnet101": resnet101,
                   "densenet121": densenet121,
                   "densenet161": densenet161,
                   "inception": inception,
                   "resnext50_32x4d": resnext50_32x4d,
                   "resnext101_32x4d": resnext101_32x4d,
    }
    return model_dict[backbone]

def build_model2(backbone):
    if backbone=="resnet18":
        model = models.resnet18(pretrained=True)
    elif backbone=='resnet34':
        model = models.resnet34(pretrained=True)
    elif backbone == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif backbone == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif backbone == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif backbone == 'inception':
        model = models.inception_v3(pretrained=True)
    elif backbone == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
    elif backbone == 'resnext101_32x4d':
        model = models.resnext101_32x8d(pretrained=True)
    elif backbone.find('efficient')!=-1:
        model = effnet.EfficientNet.from_pretrained(backbone)

    return model

class ETNet(nn.Module):
    def __init__(self, backbone, out_dim):
        super(ETNet, self).__init__()
        self.enet = build_model2(backbone)
        # self.enet.load_state_dict(torch.load(pretrained_model[backbone]))
        
        if backbone.find('efficient')!=-1:
            self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
            self.enet._fc = nn.Identity()

        else:
            self.myfc = nn.Linear(self.enet.fc.in_features, out_dim)
            self.enet.fc = nn.Identity()
#         self.conv1 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=3, bias=False)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=3, bias=False)


    def extract(self, x):
        
        return self.enet(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.extract(x)
        x = self.myfc(x)
        return x
