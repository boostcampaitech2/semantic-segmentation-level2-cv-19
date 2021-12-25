# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import argparse

import easydict
import yaml 

import torch
import torch.nn as nn
import torch.nn.functional as F

import _init_paths
import models

import torch.distributed as dist

device = 'cuda'

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    args = parser.parse_args()

    return args

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()        
        self.encoder = eval('models.'+config.MODEL.NAME +'.get_seg_model')(config)         

    #@autocast()
    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(input=x[0], size=(512, 512), mode='bilinear', align_corners=True)
        return x

def main():
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)    

    args = parse_args()

    with open(args.cfg) as f:
        config = easydict.EasyDict(yaml.load(f))

    input = torch.rand(2, 3, 512, 512).to(device)
    model =  Encoder(config).to(device)
    output = model(input)

    print(f'input.size()={input.size()}')
    print(f'output.size()={output.size()}')

if __name__ == '__main__':
    main()
