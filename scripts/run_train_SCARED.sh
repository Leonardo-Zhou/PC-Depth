export CUDA_VISIBLE_DEVICES=0

# SCARED data path
DATASET=/dataset/c3vd_train/c3vd/train

python -W ignore train.py \
    --dataset_dir $DATASET \
    --batch_size 12 \
    --dataset_name "SCARED" \
    --exp_name PC_SCARED \
    --skip_frames 1 \
    --model_version PC-Depth \
    --point_weight 0.01 \
    --smooth_weight 0.001 \
    --geometry_weight 0.1 \
    --num_epochs 20 \
    --lr 1e-4 \
    --scheduler_step_size 10 \
    --light_mu 0\
    --do_color_aug