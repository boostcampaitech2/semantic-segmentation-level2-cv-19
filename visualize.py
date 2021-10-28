
import os
import argparse
import pandas as pd
from importlib import import_module
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from seg_utils.Dataset import CustomDataLoader, collate_fn
from seg_utils.utils import plot_examples


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_colormap = pd.read_csv(args.colomap_class_dict)
    file_name = args.model_path.split('/')[-1].split('.')[0]

    # transform
    train_transform = A.Compose([ToTensorV2()])
    val_transform = A.Compose([ToTensorV2()])
    test_transform = A.Compose([ToTensorV2()])

    # dataset
    train_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='val', transform=val_transform)
    test_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='test', transform=test_transform)

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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # best model 불러오기
    model_module = getattr(import_module("seg_utils.models"), args.model)
    model = model_module()
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(device)

    plot_examples(train_loader, model, args.saved_dir, f"{file_name}_train.png", class_colormap, device, mode="train", batch_id=7, num_examples=4)
    plot_examples(val_loader, model, args.saved_dir, f"{file_name}_val.png", class_colormap, device, mode="val", batch_id=0, num_examples=4)
    plot_examples(test_loader, model, args.saved_dir, f"{file_name}_test.png", class_colormap, device, mode="test", batch_id=0, num_examples=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str, default='../input/data', help='dataset_path')
    parser.add_argument('--saved_dir', type=str, default='./vis', help='vis')
    parser.add_argument('--save_file_name', type=str, default='hrnet_w48.csv', help='save_file_name')
    parser.add_argument('--model_path', type=str, default='./saved/hrnet_w48.pt', help='model_path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 4)')
    parser.add_argument('--model', type=str, default='hrnet_w48', help='model')
    parser.add_argument('--colomap_class_dict', type=str, default="./class_dict.csv", help='colomap_class_dict')
    args = parser.parse_args()

    if not os.path.isdir(args.saved_dir):
        os.mkdir(args.saved_dir)

    main(args)