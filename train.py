import os
import numpy as np
import random
import argparse
from importlib import import_module
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from seg_utils.Dataset import CustomDataLoader, collate_fn
from seg_utils.train_validation import train


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transform
    train_transform = A.Compose([ToTensorV2()])
    val_transform = A.Compose([ToTensorV2()])

    # dataset
    train_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # model 정의
    model_module = getattr(import_module("seg_utils.models"), args.model)
    model = model_module()
    # Loss function 정의
    criterion_module = getattr(import_module("seg_utils.loss"), args.criterion)
    criterion = criterion_module()
    # Optimizer 정의
    optimizer_module = getattr(import_module("torch.optim"), 'Adam')
    optimizer = optimizer_module(params = model.parameters(), lr = args.learning_rate, weight_decay=1e-6)

    # Train
    train(args.num_epochs, 
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            args.saved_dir, 
            args.save_file_name, 
            args.val_every, 
            device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str, default='../input/data', help='dataset_path')
    parser.add_argument('--saved_dir', type=str, default='./saved', help='saved_dir')
    parser.add_argument('--save_file_name', type=str, default='fcn_resnet50_best_model(pretrained).pt', help='save_file_name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=20, help='num_epochs (default: 20)')
    parser.add_argument('--num_classes', type=int, default=11, help='num_classes (default: 11)')
    parser.add_argument('--val_every', type=int, default=1, help='val_every (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--model', type=str, default='hrnet_w48', help='model')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    args = parser.parse_args()

    # hyper parameter
    
    if not os.path.isdir(args.saved_dir):
        os.mkdir(args.saved_dir)

    main(args)