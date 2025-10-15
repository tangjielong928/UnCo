python test.py \
    --bart_name facebook/bart-base \
    --model_weight ./saved_model/best_model \
    --datapath  ./data/twitterGMNER \
    --image_feature_path ./data/Vinvl_detection_path \
    --image_annotation_path ./data/images_annotation \
    --box_num 18 \
    --batch_size 32 \
    --max_len 30 \
    --normalize \
    --mc_dropout \
    --mc_times 10  
          