python train.py \
        --save_file_name hrnet_w48_focal.pt \
        --batch_size 16 \
        --num_epochs 60 \
        --model hrnet_w48 \
        --learning_rate 1e-4 \
        --criterion FocalLoss
