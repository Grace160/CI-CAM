#!/bin/bash
#SBATCH -A test
#SBATCH -J attn_reg
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH -p short
#SBATCH -t 3-0:00:00
#SBATCH -o wetr_attn_reg.out

#source activate py36
CHECKPOINT=$1
port=29503
crop_size=512

file=scripts/dist_val_coco.py
config=configs/coco_attn_reg.yaml

echo python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port $file --config $config --pooling gmp  --work_dir work_dir_final
CUDA_VISIBLE_DEVICES=3  python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port $file --config $config --pooling gmp  --work_dir work_dir_final --checkpoint $CHECKPOINT

