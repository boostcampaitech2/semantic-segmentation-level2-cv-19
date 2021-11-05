# Semantic Segmentation Github
 
## Baseline Code: HRN OCR
- HRN(High Resolution Netork) 및 OCR(Objecct Conceptual Representation) official code를 기반으로 하였습니다.
- https://github.com/HRNet/HRNet-Semantic-Segmentation

## 설치하는 방법
- git clone https://github.com/HRNet/HRNet-Semantic-Segmentation  
- HRNet-Semantic-Segmentation-HRNet-OCR 디렉토리로 이동
- conda create --name seg_hrn_ocr --clone segmentation
- conda activate seg_hrn_ocr
- pip install -r requirements.txt   

## train하는 방법
- python tools/train.py --cfg experiments/bc_trash/hrnet_seg_all.yaml   
- cfg argumen로는 hrn ocr model을 위한 yaml 파일이 필요합니다. 


## inference 및 ensemble 하는 방법
- python tools/test.py --cfg experiments/bc_trash/hrnet_seg.yaml --model_names seg_hrnet_ocr,unetv2_b7  --pth_files  saved/best____mIoU.pth,saved/unetv2_b7.pt   --save_file submission/best_mIoU.csv     
- cfg argument로는 hrn ocr model을 위한 yaml 파일이 필요합니다. 
- model_argument로는 inferene 및 ensemble할 model명이 필요합니다. 예를 들어서 hrn ocr의 경우는 hrn_ocr이고 unet version2 /  efficientnet b7의 경우는 unetv2_b7 입니다. test.py에서 지원되는 모델을 확인할 수 있습니다.
- pth_file에는 pretrained 파일이 필요합니다. model_argument에 기록한 model마다 pretrained된 file이 필요합니다.


## test 결과 
- (1) hrn ocr seed:42 epoch:60  -> public LB mIoU: 0.675
- (2) hrn ocr seed:43 epoch:80  -> public LB mIoU: 0.692
- (3) hrn ocr seed:44 epoch:100 -> public LB mIoU: 0.676

- (2), (3), (1)을 0.5, 0.3, 0.2의 weight로 Soft Voting Ensemble -> public LB mIoU: 0.713
- (2,) (3,),(1), unet v2 with efficient net b7, unet v2 with efficient net b8을 Soft Voting Ensemble -> public LB mIoU: 0.697
- 가장 public LB score가 높은 3개 model의 ensemble 조합으로 최종 제출

 
