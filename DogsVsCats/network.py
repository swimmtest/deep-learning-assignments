import torch
import torch.nn as nn
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# custom define network
class Net(nn.Module):                                      
    def __init__(self, n_classes=2):                                     
        super().__init__()  
        self.layer1 = nn.Sequential(    
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(96))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=(2,2)),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(256))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=(1,1)),
            nn.ReLU(True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=(1,1)),
            nn.ReLU(True))

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2))

        self.layer6 = nn.Sequential(
            nn.Linear(5*5*256, 4096),
            nn.ReLU(True),
            nn.Dropout())

        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout())

        self.layer8 = nn.Linear(4096, n_classes)

        # initial parameters
        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, nn.BatchNorm2d):
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
             if isinstance(m, nn.Linear):
                 nn.init.xavier_normal_(m.weight)
                 nn.init.constant_(m.bias, 0)         

    def forward(self, x):                  
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 5*5*256)
        x = self.layer8(self.layer7(self.layer6(x)))     

        return x


# import torchvision.models as models

# def get_model(n_classes=2, pretrained=False, pretrained_path=None, vis_model=False):
#     # load pretrained parameters
#     model = models.alexnet(pretrained=pretrained)
#     n_feature = model.classifier._modules["6"].in_features
#     model.classifier._modules["6"] = nn.Linear(n_feature, n_classes)
#     # also you can print(model) when debug, modify classifier directly
#     # model.classifier[6] = nn.Linear(4096, n_classes)
    
#     # init model parameters
#     for name, param in model.named_parameters():
#         if name.endswith("weight"):
#             nn.init.xavier_normal_(param)
#         else:
#             nn.init.zeros_(param)

#     # load pretrained parameters
#     # if pretrained_path:
#     #     model.load_state_dict(torch.load(pretrained_path))

#     # visualization model
#     # if vis_model:
#     #     from torchsummary import summary
#     #     summary(model, input_size(3, 224, 224), device="cpu")

#     return model
