
import os
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from seg_utils.Dataset import CustomDataLoader, collate_fn
from seg_utils.models import fcn_resnet50
from seg_utils.train_validation import test


# best model 저장된 경로
dataset_path  = '../input/data'
model_path = './saved/fcn_resnet50_best_model(pretrained).pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
saved_dir = './submission'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)
save_file_name = 'fcn_resnet50_best_model(pretrained).csv'
batch_size = 16 

# test data loder
test_transform = A.Compose([ToTensorV2()])
test_dataset = CustomDataLoader(data_dir=dataset_path, mode='test', transform=test_transform)
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

# test set에 대한 prediction
file_names, preds = test(model, test_loader, device)

# PredictionString 대입
submission = pd.DataFrame(data=[], index=[], columns=['image_id', 'PredictionString'])
for file_name, string in tqdm(zip(file_names, preds)):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, ignore_index=True)

# submission.csv로 저장
submission.to_csv(os.path.join(saved_dir, save_file_name), index=False)