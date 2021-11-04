import os
import argparse
import pandas as pd
from importlib import import_module
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from seg_utils.Dataset import CustomDataLoader, collate_fn
from seg_utils.train_validation import test


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # test data loder
    test_transform = A.Compose([ToTensorV2()])
    test_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # best model 불러오기
    
    # model 정의
    model_module = getattr(import_module("seg_utils.models"), args.model)
    model = model_module()
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device, args.crf_mode)

    # PredictionString 대입
    submission = pd.DataFrame(data=[], index=[], columns=['image_id', 'PredictionString'])
    for file_name, string in tqdm(zip(file_names, preds)):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(args.saved_dir, args.save_file_name), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str, default='../input/data', help='dataset_path')
    parser.add_argument('--saved_dir', type=str, default='./submission', help='submission')
    parser.add_argument('--save_file_name', type=str, default='hrnet_w48.csv', help='save_file_name')
    parser.add_argument('--model_path', type=str, default='./saved/hrnet_w48.pt', help='model_path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 4)')
    parser.add_argument('--model', type=str, default='hrnet_w48', help='model')
    parser.add_argument('--crf_mode', type=bool, default=True, help='crf')
    args = parser.parse_args()

    if not os.path.isdir(args.saved_dir):
        os.mkdir(args.saved_dir)

    main(args)