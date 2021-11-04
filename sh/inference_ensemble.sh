python inference_ensemble_crf.py \
        --cfg seg_utils/hrnet_ocr/hrnet_seg.yaml \
        --model_names seg_hrnet_ocr \
        --pth_files  saved/best.pt \ 
        --save_file submission/temp1.csv

# python inference_ensemble_crf.py \
#         --cfg seg_utils/hrnet_ocr/hrnet_seg.yaml \
#         --model_names seg_hrnet_ocr,seg_hrnet_ocr,seg_hrnet_ocr,h,j \
#         --pth_files  saved/best____mIoU.pth,saved/best___mIoU.pth,saved/best__mIoU.pth,saved/han.pt,saved/jung.pt \ 
#         --save_file submission/temp1.csv