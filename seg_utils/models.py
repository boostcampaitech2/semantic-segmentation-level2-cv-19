import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import segmentation_models_pytorch as smp


# ====================================================================================
# ================================== fcn_resnet50 ====================================
# ====================================================================================
class fcn_resnet50(nn.Module):
    '''
    fcn_resnet50
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)

# ====================================================================================
# ============================== efficientnet_b0_Unet ================================
# ====================================================================================
class efficientnet_b0(nn.Module):
    def __init__(self):
        super().__init__()
        model = smp.Unet(encoder_name="efficientnet-b0", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=11,)                    # model output channels (number of classes in your dataset)

    def forward(self, x):
        return self.model(x)



# ====================================================================================
# ================================== hrnet_unet ======================================
# ====================================================================================
class hrnet_w48(nn.Module):
    '''
    AI stage 토론 게시판
    https://stages.ai/competitions/78/discussion/talk/post/809
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="tu-hrnet_w48",       
            encoder_weights="imagenet",   
            in_channels=3,                
            classes=11,)

    def forward(self, x):
        return self.model(x)
