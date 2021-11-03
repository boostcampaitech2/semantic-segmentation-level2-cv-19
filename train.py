import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
from losses import DiceCELoss
import argparse
import pickle
import segmentation_models_pytorch as smp

from dataset import *
from train_valid import *

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    
def main():
    '''
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    parser.add_argument('--random_seed', default=21, type=int)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    random_seed=21
    num_epochs = 60
    batch_size = 5
    learning_rate = 0.0001
    dataset_path  = '../input/data'
    anns_file_path = dataset_path + '/' + 'train_all.json'
    
    fix_seed(random_seed)
    
    # train.json / validation.json / test.json 디렉토리 설정
    train_path = dataset_path + '/train_all.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'
    
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    #https://gist.github.com/ernestum/601cdf56d2b424757de5 elastic transform
    train_transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.OneOf([A.RandomRotate90(p=1.0), A.Rotate(limit=[-30,30],p=1)]),
                                ToTensorV2()
                                ])
    
    val_transform = A.Compose([
                              ToTensorV2()
                              ])
    
    test_transform = A.Compose([
                               ToTensorV2()
                               ])
    
    # train dataset
    with open("classdict.pickle","rb") as cld:
        classdict = pickle.load(cld)
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform,augmix = classdict)
    
    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
    
    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    
    
    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              collate_fn=collate_fn)
    
    # setting model
    model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b8",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=11,  # model output channels (number of classes in your dataset)
    )
    
    # 모델 저장 함수 정의
    val_every = 1

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    # best model 저장된 경로
    model_path = './saved/unetpp(pretrained).pt'

    # Loss function 정의
    criterion = DiceCELoss()
    
    logger = logging.getLogger("Segmentation")
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, f'unetpp 211103.log'))
    logger.addHandler(file_handler)
    
    # Optimizer 정의
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
    
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer,logger, saved_dir, val_every, device)

if __name__ == "__main__":
    main()