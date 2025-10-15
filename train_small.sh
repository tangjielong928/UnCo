
python train.py \
    --bart_name bart-base \
    --n_epochs 30 \
    --seed 42 \
    --datapath  ./data/twitterGMNER \
    --image_feature_path ./data/Vinvl_detection_path \
    --image_annotation_path ./data/images_annotation \
    --lr 3e-5 \
    --box_num 18 \
    --batch_size 32 \
    --max_len 30 \
    --save_model 1 \
    --normalize \
    --use_kl \
    --save_path ./saved_model/best_model \
    --log ./logs/ \
    --triplet_weight 0.6 
    