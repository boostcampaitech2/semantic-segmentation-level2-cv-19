import os
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from collections import defaultdict

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.optim.lr_scheduler import CosineAnnealingLR

import _init_paths
import models
import datasets

import easydict
import yaml 

import torch.distributed as dist
import json
import pandas as pd

from utils.baseline_utils import label_accuracy_score, add_hist

from core.criterion import DiceWCELoss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_model(model, save_type='loss'):
    save_path = os.path.join(f'./saved')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'best_{save_type}.pth')
    torch.save(model.state_dict(), save_dir)
    
    
def get_categories(train_anno_path):
    with open(train_anno_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    print('Number of super categories:', nr_super_cats)
    print('Number of categories:', nr_cats)
    print('Number of annotations:', nr_annotations)
    print('Number of images:', nr_images)
    # Count annotations
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1

#    # Initialize the matplotlib figure
#    f, ax = plt.subplots(figsize=(5,5))

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
    print(sorted_df)
    return sorted_df    
    
    
def train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, scheduler, device, beta=0.0):
    cnt = 1

    model.train()
    trn_losses = []

    hist = np.zeros((11, 11))

    with tqdm(trn_dl, total=len(trn_dl), unit='batch') as trn_bar:
        for batch, sample in enumerate(trn_bar):
            trn_bar.set_description(f"Train Epoch {epoch+1}")

            optimizer.zero_grad()
            images, masks = sample['image'], sample['mask']
            images, masks = images.to(device), masks.to(device).long()

            preds = model(images)
            if isinstance(preds, list):
                for i in range(len(preds)):
                    pred = preds[i]
                    ph, pw = pred.size(2), pred.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        pred = F.interpolate(input=pred, size=(
                            h, w), mode='bilinear', align_corners=True)
                    preds[i] = pred
            
                loss = 0               
                for i in range(len(preds)):
                    loss += criterion(preds[i], masks)
                preds = preds[0]

            else:
                loss = criterion(preds, masks)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=11)

            trn_mIoU = label_accuracy_score(hist)[2]
            trn_losses.append(loss.item())

            trn_bar.set_postfix(trn_loss=np.mean(trn_losses),
                                trn_mIoU=trn_mIoU)


    model.eval()
    val_losses = []
    hist = np.zeros((11, 11))

    example_images = []


    with torch.no_grad():
        with tqdm(val_dl, total=len(val_dl), unit='batch') as val_bar:
            for batch, sample in enumerate(val_bar):
                val_bar.set_description(f"Valid Epoch {epoch+1}")

                images, masks = sample['image'], sample['mask']
                images, masks = images.to(device), masks.to(device).long()

                preds = model(images)
                if isinstance(preds, list):
                    _, preds = preds
                    ph, pw = preds.size(2), preds.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        preds = F.interpolate(input=preds, size=(
                            h, w), mode='bilinear', align_corners=True)

                loss = criterion(preds, masks)
                val_losses.append(loss.item())

                preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=11)
                _, _, val_mIoU, _, val_IoU = label_accuracy_score(hist) 
                val_bar.set_postfix(val_loss=np.mean(val_losses),
                                    val_mIoU=val_mIoU)
 
    return np.mean(trn_losses), trn_mIoU, np.mean(val_losses), val_mIoU, val_IoU

def main(args):
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)    
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.cfg) as f:
        config = easydict.EasyDict(yaml.load(f))    
        
    # transform
    train_transform = A.Compose([

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        A.Normalize(),
        ToTensorV2()
        ])

    val_transform = A.Compose([
        A.Normalize(),        
        ToTensorV2()
    ])

    
    train_categories = get_categories(config.DATASET.TRAIN_SET)
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(config.DATASET.ROOT, config.DATASET.TRAIN_SET,      train_categories, mode='train', transform=train_transform)
    val_dataset   = eval('datasets.'+config.DATASET.DATASET)(config.DATASET.ROOT, config.DATASET.VALIDATION_SET, train_categories, mode='valid', transform=val_transform)    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4)

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config, mode='train_validation').to(device)
      
#    criterion = nn.CrossEntropyLoss()
#    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
#    scheduler = None    

    weights = torch.FloatTensor([0.3034, 0.9775, 0.9091, 0.9930, 0.9911, 0.9927, 0.9715, 0.9851, 0.8823, 0.9996, 0.9947]).to(device) if args.use_weight else None
    print(f'{weights}')        
    criterion = DiceWCELoss(weight=weights)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    best_loss = float("INF")
    best_mIoU = 0
    for epoch in range(args.num_epochs):
        trn_loss, trn_mIoU, val_loss, val_mIoU, _ = train_valid(epoch, model, train_loader, val_loader,
                                                             criterion, optimizer, scheduler,device)
        save_model(model, save_type='current')
        if best_loss > val_loss:
            best_loss = val_loss
            save_model(model, save_type='loss'+ str(best_loss) + '_' + str(epoch))

        if best_mIoU < val_mIoU:
            best_mIoU = val_mIoU
            save_model(model, save_type='mIoU'+ str(best_mIoU) + '_' + str(epoch))    
    





    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')    
    parser.add_argument('--saved_dir', type=str, default='./saved', help='saved_dir')   
    parser.add_argument('--num_epochs', type=int, default=20, help='num_epochs (default: 20)')    
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training (default: 4)')    
    parser.add_argument('--use_weight', default=1, type=int)    
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate (default: 1e-3)')   
    parser.add_argument('--decay', default=1e-6, type=float)        
    parser.add_argument('--cfg', type=str, required=True, help='config file name')
    
    args = parser.parse_args()    
    
    if not os.path.isdir(args.saved_dir):
        os.mkdir(args.saved_dir)

    main(args)
    