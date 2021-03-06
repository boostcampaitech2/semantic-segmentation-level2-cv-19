import os, random, cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))
    
class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        # mode is train, val, test or train_all
        self.coco = COCO( f"{data_dir}/{self.mode}.json")
        self.category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal','Glass', 
                                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    def get_categoties(self):
        pass
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
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
            # General trash = 1, ... , Cigarette = 10
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
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

    def augmix_search(self, images, masks):
        # image 3, 512, 512 ,mask: 512, 512 (둘 다 numpy)
        tfms = A.Compose([
                    # A.Resize(384, 384, p=1.0)
                    A.GridDistortion(p=0.3, distort_limit=[-0.01, 0.01]),
                    A.Rotate(limit=60, p=1.0),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5)
              ])
        
        num = [3, 4, 5, 9, 10]

        label = random.choice(num)  # ex) 4
        idx = np.random.randint(len(self.augmix[label]))
        augmix_img = self.augmix[label][idx]
        augmix_mask = np.zeros((512, 512))
        # augmix img가 있는 만큼 label로 mask를 채워줌
        augmix_mask[augmix_img[:, :, 0] != 0] = label
        ################################################## 새로 추가한 transform을 적용해보자 
        transformed=tfms(image=augmix_img, mask=augmix_mask)
        augmix_img = transformed['image']
        augmix_mask = transformed['mask']
        ####################################################
        images[augmix_img != 0] = augmix_img[augmix_img != 0]
        masks[augmix_mask != 0] = augmix_mask[augmix_mask != 0]

        return images, masks


class CustomAugmentation:
    def __init__(self, mode = 'train'):
        if mode == 'train':
            self.transform = A.Compose([
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5),
                                    A.Normalize(),
                                    ToTensorV2()
                                    ])
        elif mode == 'val':
            self.transform = A.Compose([
                                    A.Normalize(),        
                                    ToTensorV2()
                                    ])
        else:
            self.transform = A.Compose([
                                    ToTensorV2()
                                    ])

    def __call__(self, image, mask=None):
        if mask==None:
            return self.transform(image=image)
        else:
            return self.transform(image=image, mask=mask)


if __name__=="__main__":
    dataset_path  = '../input/data'
    anns_file_path = dataset_path + '/' + 'train_all.json'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    train_dataset = CustomDataLoader(data_dir=dataset_path, mode='train', transform=None)