import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import segmentation_models_pytorch as smp
# Hrnet OCR 
import yaml 
from seg_utils.hrnet_ocr.seg_hrnet_ocr import get_seg_model

# ================================== fcn_resnet50 ====================================
class FcnResnet50(nn.Module):
    '''
    fcn_resnet50
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# ============================== efficientnet_b0_Unet ================================
class EfficientnetB0(nn.Module):
    def __init__(self):
        super().__init__()
        model = smp.Unet(encoder_name="efficientnet-b0", 
                        encoder_weights="imagenet",     
                        in_channels=3,                  
                        classes=11,)                    

    def forward(self, x):
        return self.model(x)



# ================================== hrnet_unet ======================================
class HrnetW48(nn.Module):
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


# ================================== Unet++ efficientnet-b8 ======================================
class UnetPlusPlusB8(nn.Module):
    '''
    AI stage 토론 게시판
    https://stages.ai/competitions/78/discussion/talk/post/809
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b8",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            )
    def forward(self, x):
        return self.model(x)


# ================================== Unet++ efficientnet-b7 ======================================
class UnetPlusPlusB7(nn.Module):
    '''
    AI stage 토론 게시판
    https://stages.ai/competitions/78/discussion/talk/post/809
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,  
            classes=11,  
        )
    def forward(self, x):
        return self.model(x)


# ================================== hrnet ocr ======================================
# AssertionError: Default process group is not initialized
# solution: https://github.com/pytorch/pytorch/issues/22538
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
    
class HrnetOcr(nn.Module):
    '''
    https://stages.ai/competitions/78/discussion/talk/post/809 2.참고
    '''
    def __init__(self):
        super().__init__()        
        config_path='./seg_utils/hrnet_ocr/hrnet_seg.yaml'
        with open(config_path) as f:
            cfg = yaml.load(f)
        self.encoder = get_seg_model(cfg)
        
    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(input=x[0], size=(512, 512), mode='bilinear', align_corners=True)
        return x
