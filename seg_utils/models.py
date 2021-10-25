import torch.nn as nn
from torchvision import models

class fcn_resnet50(nn.Module):
    '''
    fcn_resnet50
    '''
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)

