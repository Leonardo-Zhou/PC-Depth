# absolute path that contains all datasets
DATA_ROOT=/dataset

# # kitti
# DATASET=$DATA_ROOT/kitti
# CONFIG=configs/v1/kitti_raw.txt

# # nyu
# DATASET=$DATA_ROOT/nyu
# CONFIG=configs/v2/nyu.txt

# ddad
DATASET=/dataset/c3vd_train/c3vd/train

export CUDA_VISIBLE_DEVICES=0

# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name SC \
#     --lr 1e-4 \
#     --skip_frames 10 \
#     --model_version SC-Depth \
#     --point_weight 0.0 \
#     --block_weight 0.0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0.0 \
#     --geometry_weight 0.1


# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name IA_final \
#     --skip_frames 10 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10\
#     --do_color_aug

# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name IA_final \
#     --skip_frames 10 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10\
#     --do_color_aug\
#     --no_inpaint_smooth

# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name IA_final \
#     --skip_frames 10 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10\
#     --do_color_aug\
#     --no_inpaint_smooth\
#     --no_light_align


# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name no_LP \
#     --skip_frames 10 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10\
#     --do_color_aug\
#     --no_light_align\
#     --no_specular_mask

python -W ignore train.py \
    --dataset_dir $DATASET \
    --batch_size 4 \
    --exp_name test \
    --skip_frames 10 \
    --model_version PC-Depth \
    --point_weight 0.001 \
    --block_weight 0 \
    --smooth_weight 0.01 \
    --k_smooth_weight 0 \
    --geometry_weight 0.1 \
    --num_epochs 20 \
    --lr 1e-4 \
    --scheduler_step_size 10\
    --do_color_aug


# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name   \
#     --skip_frames 10 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10\
#     --do_color_aug\
#     --no_specular_mask

# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 4 \
#     --exp_name HS_new \
#     --skip_frames 10 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0.1 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10 \
#     --epoch_size 1650\
#     --no_light_align



# DATASET=/dataset/SCARED_train/
# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 12 \
#     --dataset_name "SCARED" \
#     --exp_name SC_SCARED \
#     --skip_frames 1 \
#     --model_version SC-Depth \
#     --point_weight 0.1 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10 \
#     --epoch_size 1300 \
#     --light_mu 0

# DATASET=/dataset/SCARED_train/
# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 12 \
#     --dataset_name "SCARED" \
#     --exp_name IA_SCARED \
#     --skip_frames 1 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10 \
#     --light_mu 0

# DATASET=/dataset/SCARED_train/
# python -W ignore train.py \
#     --dataset_dir $DATASET \
#     --batch_size 12 \
#     --dataset_name "SCARED" \
#     --exp_name IA_SCARED \
#     --skip_frames 1 \
#     --model_version PC-Depth \
#     --point_weight 0.01 \
#     --block_weight 0 \
#     --smooth_weight 0.1 \
#     --k_smooth_weight 0 \
#     --geometry_weight 0.1 \
#     --num_epochs 20 \
#     --lr 1e-4 \
#     --scheduler_step_size 10