export CUDA_VISIBLE_DEVICES=0

# Folder to store train sequence
DATASET=/dataset/c3vd_train/c3vd/train

python -W ignore train.py \
    --dataset_dir $DATASET/ \
    --batch_size 4 \
    --dataset_name "c3vd" \
    --exp_name C3VD \
    --skip_frames 10 \
    --model_version PC-Depth \
    --point_weight 0.01 \
    --smooth_weight 0.001 \
    --geometry_weight 0.1 \
    --num_epochs 20 \
    --lr 1e-4 \
    --scheduler_step_size 10\
    --light_mu 3.069096\
    --do_color_aug