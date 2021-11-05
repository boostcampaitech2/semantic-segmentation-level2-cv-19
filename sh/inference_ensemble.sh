python inference_ensemble_crf.py \
        --cfg seg_utils/hrnet_ocr/hrnet_seg.yaml \
        --model_names seg_hrnet_ocr,seg_hrnet_ocr \
        --pth_files  saved/hrnet_w48.pt,saved/hrnet_w48.pt \
        --save_file submission/temp1.csv

