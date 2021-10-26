
import os
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from seg_utils.Dataset import CustomDataLoader, collate_fn
from seg_utils.models import fcn_resnet50
from seg_utils.train_validation import test

from seg_utils.utils import plot_examples

# best model 저장된 경로
dataset_path  = '../input/data'
model_path = './saved/fcn_resnet50_best_model(pretrained).pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
saved_dir = './vis'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)
colomap_class_dict = "./class_dict.csv"
batch_size = 16 

class_colormap = pd.read_csv(colomap_class_dict)
file_name = model_path.split('/')[-1].split('.')[0]

# transform
train_transform = A.Compose([ToTensorV2()])
val_transform = A.Compose([ToTensorV2()])
test_transform = A.Compose([ToTensorV2()])

# dataset
train_dataset = CustomDataLoader(data_dir=dataset_path, mode='train', transform=train_transform)
val_dataset = CustomDataLoader(data_dir=dataset_path, mode='val', transform=val_transform)
test_dataset = CustomDataLoader(data_dir=dataset_path, mode='test', transform=test_transform)

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

# best model 불러오기
model = fcn_resnet50()
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint.state_dict()
model.load_state_dict(state_dict)
model = model.to(device)

plot_examples(train_loader, model, saved_dir, f"{file_name}_train.png", class_colormap, device, mode="train", batch_id=7, num_examples=4)
plot_examples(val_loader, model, saved_dir, f"{file_name}_val.png", class_colormap, device, mode="val", batch_id=0, num_examples=4)
plot_examples(test_loader, model, saved_dir, f"{file_name}_test.png", class_colormap, device, mode="test", batch_id=0, num_examples=8)