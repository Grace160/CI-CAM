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

port=29501
crop_size=512
queue_len=300
momentum=0.99

file=scripts/dist_train_voc.py
config=configs/voc_attn_reg.yaml

echo python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final l --queue_len $queue_len  --momentum $momentum
CUDA_VISIBLE_DEVICES=0,1  python -u -m torch.distributed.launch --nproc_per_node=2  --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final --queue_len $queue_len  --momentum $momentum

