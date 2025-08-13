export CUDA_VISIBLE_DEVICES=0

# SCARED data path
DATASET=/mnt/data/publicData/MICCAI19_SCARED/train

python -W ignore train.py \
    --dataset_dir $DATASET \
    --batch_size 12 \
    --dataset_name "SCARED" \
    --exp_name PC_SCARED \
    --skip_frames 1 \
    --model_version PC-Depth \
    --smooth_weight 0.001 \
    --geometry_weight 0.1 \
    --num_epochs 20 \
    --lr 1e-4 \
    --scheduler_step_size 10 \
    --light_mu 0\
    --do_color_aug\
    --highlight_weight=0.01