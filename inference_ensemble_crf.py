import os, argparse
import numpy as np
from tqdm import tqdm
import easydict, yaml
import ttach as tta

import torch
import torch.nn.functional as F
import torch.distributed as dist
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import multiprocessing as mp
import numpy as np
import pandas as pd
from seg_utils.utils import seed_everything, dense_crf_wrapper, make_cat_df
from seg_utils.Dataset import CustomAugmentation, CustomDataLoader


def main(args):
    # dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)        
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(args.saved_dir):                                                           
        os.mkdir(args.saved_dir)
    save_file_name = args.save_file
    batch_size = args.batch_size
    test_annot = os.path.join(args.dataset_path, 'test.json')    

    with open(args.cfg) as f:
        config = easydict.EasyDict(yaml.load(f))    

    # test data loder
    test_transform = CustomAugmentation('val')
    test_dataset = CustomDataLoader(data_dir=args.dataset_path, mode='val', transform=test_transform)        
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=4
                                        )    
    model_names = args.model_names.split(',')
    pths = args.pth_files.split(',')
    
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

        elif model_name == 'h':
            model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,  
            classes=11, 
            )                       
            checkpoint = torch.load(pth, map_location=device)
            state_dict = checkpoint.state_dict()
            model.load_state_dict(state_dict)    
                    
        elif model_name == 'j':
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

        tta_tfms = tta.Compose([tta.VerticalFlip(),                                
                                tta.HorizontalFlip(),
                                tta.Rotate90([0, 90]),
                                ])

        model = tta.SegmentationTTAWrapper(model, tta_tfms, merge_mode='mean')
        
            
        model = model.to(device)    
        test_models.append(model)

    print(weights[len(test_models)])
        
    tar_size = 512
    size = 256
    resize = A.Resize(size, size)

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)
        
    resize_transform = A.Compose([A.Resize(size, size)])
    with torch.no_grad():
        for step, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            imgs = sample['image']
            file_names = sample['info']

            final_probs = 0            
            for idx, (model, weight) in enumerate(zip(test_models, weights[len(test_models)])):
                imgs = imgs.to(device)
                preds = model(imgs)

                if isinstance(preds, list):
                    preds = preds[0]
                ph, pw = preds.size(2), preds.size(3)
                if ph != 512 or pw != 512:
                    preds = F.interpolate(input=preds, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
                probs = F.softmax(preds, dim=1).detach().cpu().numpy()

                if args.crf_mode==True:
                    pool = mp.Pool(mp.cpu_count())
                    images = imgs.detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                    if images.shape[1] != tar_size or images.shape[2] != tar_size:
                        images = np.stack([resize(image=im)['image'] for im in images], axis=0)
                    probs = np.array(pool.map(dense_crf_wrapper, zip(images, probs)))

                final_probs += weight * probs
                pool.close()
            oms = np.argmax(final_probs.squeeze(), axis=1)

            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()  # batch_size, channel, h, w -> batch_size, h, w, channel
            for img, mask in zip(temp_images, oms):
                if mask.shape[0] != 256 or mask.shape[1] != 256:
                    transformed = resize_transform(image=img, mask=mask)
                    mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([file_name for file_name in file_names])
            #break
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
    parser.add_argument('--batch_size', type=int, default=16)        
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str )
    parser.add_argument('--model_names', type=str, default='seg_hrnet_ocr,seg_hrnet_ocr', help='model names')    
    parser.add_argument('--pth_files', type=str, default='./saved/best_mIoU.pth,./saved/best_loss.pth', help='trained model files')                           
    parser.add_argument('--save_file',  type=str, default='./submission/best_mIOU.csv', help='submission file')                           
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/segmentation/input/data', help='trained model files')                           
    parser.add_argument('--saved_dir', type=str, default='./submission', help='trained model files')                           
    parser.add_argument('--crf_mode',  type=bool, default=False, help='crf mode')                           
    args = parser.parse_args()        
    main(args)