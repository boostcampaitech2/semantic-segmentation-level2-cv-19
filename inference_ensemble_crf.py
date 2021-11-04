import os, argparse, random, json
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
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import numpy as np
import pandas as pd

def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])

def dense_crf(img, output_probs):
    '''
    https://www.programcreek.com/python/example/106424/pydensecrf.densecrf.DenseCRF2D
    '''
    MAX_ITER = 10
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q
            
            
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"

def make_cat_df(train_annot_path, debug=False):
    with open(train_annot_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    print('Number of super categories:', nr_super_cats)
    print('Number of categories:', nr_cats)
    print('Number of annotations:', nr_annotations)
    print('Number of images:', nr_images)
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
    return sorted_df    
    

def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
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
    seed_everything(args.seed)

    with open(args.cfg) as f:
        config = easydict.EasyDict(yaml.load(f))    

    dataset_path  = '/opt/ml/segmentation/input/data'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved_dir = './submission'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
    save_file_name = args.save_file
    batch_size = 16 

    test_annot = os.path.join(dataset_path, 'test.json')    
    test_cat = make_cat_df(test_annot, debug=True)    

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
    #print(model_names)
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

                ph, pw = preds.size(2), preds.size(3)
                if ph != 512 or pw != 512:
                    preds = F.interpolate(input=preds, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
                probs = F.softmax(preds, dim=1).detach().cpu().numpy()

                pool = mp.Pool(mp.cpu_count())
                images = imgs.detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                if images.shape[1] != tar_size or images.shape[2] != tar_size:
                    images = np.stack([resize(image=im)['image'] for im in images], axis=0)

                final_probs += np.array(pool.map(dense_crf_wrapper, zip(images, probs))) / 5
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
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str )
    parser.add_argument('--model_names', type=str, default='seg_hrnet_ocr,seg_hrnet_ocr', help='model names')                           
    parser.add_argument('--pth_files', type=str, default='./saved/best_mIoU.pth,./saved/best_loss.pth', help='trained model files')                           
    parser.add_argument('--save_file',  type=str, default='./submission/best_mIOU.csv', help='submission file')                           
    args = parser.parse_args()        
    main(args)