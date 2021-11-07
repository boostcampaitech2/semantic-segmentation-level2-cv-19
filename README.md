# Semantic Segmentation for classifying recycling item

## 1. Introduction

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.  
분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.  
따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.  
여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎  


## 2. Dataset

**재활용 쓰레기 데이터셋 / Aistages(Upstage) - CC BY 2.0**  

- 전체 이미지 개수: 3272장
- 11 class: Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기: (512, 512)

### Class Description
| class | trash |
|---|:-------------:|
| 0 | Background    |
| 1 | General trash |
| 2 | Paper         |
| 3 | Paper pack    |
| 4 | Metal         |
| 5 | Glass         |
| 6 | Plastic       |
| 7 | Styrofoam     |
| 8 | Plastic bag   |
| 9 | Battery       |
| 10 | Clothing      |

### Dataset folder path
```
 input
    └── data
        ├── batch_01_vt
        ├── batch_02_vt
        ├── batch_03
        │
        ├── test.json
        ├── train.json
        ├── train_all.json
        └── val.json
```

## 3. Prerequisites

### Dependencies
* albumentations==1.0.3
* pycocotools==2.0.2
* opencv-python==4.5.3.56
* tqdm==4.62.3
* pandas==1.3.3
* map-boxes==1.0.5
* pytorch==1.7.1


### additional requirements in Segmentation server of AI stage
* 추가 requirements 설치
```
sh sh/install_requirements.sh
```

## 4. Train
*  HRNet OCR pretrained model download
```
mkdir pretrained
cd pretrained
wget https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W48_C_ssld_pretrained.pth
```
```
sh sh/train_hrnet.sh
```
* Model

|Model|backbone|model class|
|---|---|---|
|HRN OCR|hrnet_w48|HrnetOcr|
|Unet++|EfficientNet b7|UnetPlusPlusB7|
|Unet++|EfficientNet b8|UnetPlusPlusB8|


## 5. Test
```
sh sh/test_hrnet.sh
```

## 6. inference using ensemble and crf
```
sh sh/inference_ensemble.sh
```

## 7. Result
|Model|backbone|mIOU|TTA|crf|
|---|---|---|---|---|
|Unet++|EfficientNet-b8|0.643|o|x|
|Unet++|EfficientNet-b7|0.654|x|x|
|Unet++|EfficientNet-b7|0.677|x|x|
|HRN-OCR|hrnet_w48|0.698|o|x|
|Ensemble (3 HRN-OCR)|-|0.713|-|-|
|Ensemble (3 HRN-OCR + 2Unet++)|-|0.699|-|-|
