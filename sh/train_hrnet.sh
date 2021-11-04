python train.py \
        --save_file_name hrnet_w48_focal2.pt \
        --batch_size 16 \
        --num_epochs 60 \
        --model HrnetOcr \
        --learning_rate 1e-3 \
        --criterion FocalLoss

