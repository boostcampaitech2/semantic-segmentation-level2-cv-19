
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from seg_utils.Dataset import CustomDataLoader, collate_fn
from seg_utils.models import fcn_resnet50
from seg_utils.train_validation import train

# hyper parameter
batch_size = 16 
num_epochs = 20
learning_rate = 0.0001
dataset_path  = '../input/data'
saved_dir = './saved'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)
save_file_name = 'fcn_resnet50_best_model(pretrained).pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
val_every = 1


# transform
train_transform = A.Compose([ToTensorV2()])
val_transform = A.Compose([ToTensorV2()])

# dataset
train_dataset = CustomDataLoader(data_dir=dataset_path, mode='train', transform=train_transform)
val_dataset = CustomDataLoader(data_dir=dataset_path, mode='val', transform=val_transform)

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

# model 정의
model = fcn_resnet50()
# Loss function 정의
criterion = nn.CrossEntropyLoss()
# Optimizer 정의
optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, save_file_name, val_every, device)