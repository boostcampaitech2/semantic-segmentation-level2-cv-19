import os
import json

from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import pandas as pd

from pycocotools.coco import COCO

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"

class CustomDataset(Dataset):
    """COCO format"""

    def __init__(self, data_dir_path, annotation_file_path, dataframe_categories, mode='train', num_cls=11, transform=None):
        super().__init__()
        self.mode = mode
        self.num_cls = num_cls
        self.transform = transform
        self.coco = COCO(annotation_file_path)
        self.data_path = data_dir_path     
        self.category_names = list(dataframe_categories.Categories)
        

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if (self.mode in ('train', 'valid')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
#X ???           
#            for i in range(len(anns)):
#                className = get_classname(anns[i]['category_id'], cats)
#                pixel_value = self.category_names.index(className)
#                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
#            masks = masks.astype(np.float32)
#O ???
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
            
            
            
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return {
                'image': images,
                'mask' : masks,
                'info' : image_infos['file_name']
            }

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return {
                'image': images,
                'info': image_infos['file_name']
            }

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


class CustomDataset3(Dataset):
    """COCO format"""
#    def __init__(self, data_dir, cat_df, mode='train', num_cls=11, transform = None):
    def __init__(self, data_dir, cat_df, mode, num_cls=11, transform = None):

        super().__init__()
                
        #self.mode = mode
        self.num_cls = num_cls
        #self.transform = transform
        self.coco = COCO(data_dir)
        self.ds_path = f'{os.sep}'.join(data_dir.split(os.sep)[:-1])
        self.category_names = list(cat_df.Categories)

        self.mode = mode
        self.transform = transform
        self.data_path = self.ds_path
        self.annotation_path  = data_dir
        self.coco = COCO(self.annotation_path)
        self.category_names = self.generate_category_names()

        
    def __getitem__(self, index: int):        
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.ds_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))

            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return {
                'image': images,
                'mask' : masks,
                'info' : image_infos['file_name']
            }

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return {
                'image': images,
                'info': image_infos['file_name']
            }
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
    
    def generate_category_names(self):
        dataset_path  = self.data_path
        anns_file_path = self.annotation_path

        # Read annotations
        with open(anns_file_path, 'r') as f:
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

        # Convert to DataFrame
        df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
        df = df.sort_values('Number of annotations', 0, False)        

        # category labeling 
        sorted_temp_df = df.sort_index()

        # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
        sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
        sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)        
        print(sorted_df)
        
        return list(sorted_df.Categories)

    
if __name__ == '__main__':
    data_path = '/opt/ml/segmentation/input/data'
    train_annotation_path = data_path + '/train.json'    
    dataset = CustomDataLoader(data_path, train_annotation_path, mode='train', transform=None)

    print('files:')
    for i, (image, mask, image_info) in enumerate(dataset):
        print(image_info['file_name'])
        if i == 10:
            break
