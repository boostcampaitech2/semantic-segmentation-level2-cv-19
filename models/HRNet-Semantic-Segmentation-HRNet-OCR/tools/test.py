import os
import numpy as np
import argparse
import random

from tqdm import tqdm

import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import _init_paths
import models
import datasets

import easydict
import yaml 

import torch.distributed as dist

import ttach as tta

import segmentation_models_pytorch as smp



from utils.baseline_utils import set_seed, get_categories
    

    

def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            # outs = model(torch.stack(imgs).to(device))['out']
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array    
    
def main(args):
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)        

    set_seed(args.seed)

    with open(args.cfg) as f:
        config = easydict.EasyDict(yaml.load(f))    

        
    # best model 저장된 경로
    dataset_path  = '/opt/ml/segmentation/input/data'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved_dir = './submission'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
    save_file_name = args.save_file
    batch_size = 16 

    test_annot = os.path.join(dataset_path, 'test.json')    
    test_cat = get_categories(test_annot)    
    # test data loder
    test_transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(config.DATASET.ROOT, config.DATASET.TEST_SET, test_cat, mode='test', transform=test_transform)        

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4
    )    
    model_names = args.model_names.split(',')
    pths = args.pth_files.split(',')
    print(pths)
    
    weights = {
        2: [0.5, 0.5],
        3: [0.5, 0.3, 0.2],
        4: [0.5, 0.2, 0.2, 0.1],
        5: [0.5, 0,2, 0,2, 0.05, 0.05]
    }
          
    test_models = []
    for model_name, pth in zip(model_names, pths):
        print(f'model:{model_name} pth:{pth}')
        if model_name == config.MODEL.NAME:  #seg_hrn_ocr case
            model = eval('models.'+config.MODEL.NAME +'.get_seg_model')(config, mode='test')
            model.load_state_dict(torch.load(pth))        

        elif model_name == 'unetv2_b7':
            model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=11,  # model output channels (number of classes in your dataset)
            )                       
            checkpoint = torch.load(pth, map_location=device)
            state_dict = checkpoint.state_dict()
            model.load_state_dict(state_dict)            
        elif model_name == 'unetv2_b8':
            model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b8",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11,
            )            
            checkpoint = torch.load(pth, map_location=device)
            state_dict = checkpoint.state_dict()
            model.load_state_dict(state_dict)            
            
        else:
            print(f'Unsupported Mode:{model_name}')
            return

        tta_tfms = tta.Compose(
                            [
                                tta.VerticalFlip(),                                
                                tta.HorizontalFlip(),
                                tta.Rotate90([0, 90]),
                            ]
                        )

        model = tta.SegmentationTTAWrapper(model, tta_tfms, merge_mode='mean')
        
            
        model = model.to(device)    
        test_models.append(model)

    print(weights[len(test_models)])
        
    tar_size = 512
    size = 256
    
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)
        
    resize_transform = A.Compose([A.Resize(size, size)])

    with torch.no_grad():
        for step, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            imgs = sample['image']
            file_names = sample['info']

            final_probs = 0            
            for i, (model, weight) in enumerate(zip(test_models, weights[len(test_models)])):
               
                # inference (512 x 512)
                imgs = imgs.to(device)
                #print(f'img.size() {imgs.size()} ')
                preds = model(imgs)
                #print(f'preds.size() {preds.size()} ')
                if isinstance(preds, list):
                    preds = preds[0]
                ph, pw = preds.size(2), preds.size(3)
                if ph != tar_size or pw != tar_size:
                    preds = F.interpolate(input=preds, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
                        
                #print(preds.size()) # torch batch_size, channel, h , w (16, 11, 512, 512)
                probs = F.softmax(preds, dim=1)
                #print(probs.shape) # numpy batch_size, channel, h , w   (16, 11, 512, 512)
#                final_probs += probs / len(test_models)
                final_probs += probs * weight

#            oms = torch.argmax(preds, dim=1).detach().cpu().numpy()    
            oms = torch.argmax(final_probs, dim=1).detach().cpu().numpy()    

            #print(oms.shape) # numpy batch_size, channel, h , w      (16, 512, 512)
            #print(imgs.size())                                       (16, 3, 512, 512)
            # resize (256 x 256)
            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()  # batch_size, channel, h, w -> batch_size, h, w, channel
            #print(temp_images.shape)                                  (16, 512, 512, 3)
            for img, mask in zip(temp_images, oms):
                if mask.shape[0] != 256 or mask.shape[1] != 256:
                    transformed = resize_transform(image=img, mask=mask)
                    mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([file_name for file_name in file_names])
    print("End prediction.")
            
    print("Saving...")
    file_names = [y for x in file_name_list for y in x]
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    save_path = './submission'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    submission.to_csv(save_file_name, index=False)
    print("All done.")            
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')        
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str )
    parser.add_argument('--model_names', type=str, default='seg_hrnet_ocr,seg_hrnet_ocr', help='model names')                           
    parser.add_argument('--pth_files', type=str, default='./saved/best_mIoU.pth,./saved/best_loss.pth', help='trained model files')                           
    parser.add_argument('--save_file',  type=str, default='./submission/best_mIOU.csv', help='submission file')                           
    args = parser.parse_args()        
    main(args)