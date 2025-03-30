#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-302 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-12:00:00
#SBATCH -o "/mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/RGB-NIR-rPPG/log/nir_vit_snr_fold4_hr"
# sbatch --array=1-5 ./config/vit/train.sh ./config/vit/train.txt

echo "STARTING..."

input_file=$1
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

echo "-------------------------- $fold --------------------------"

# name="CPG-snr-vit-$fold-lr"

name="nir-snr-fold$fold-ft"

# name="full_nir-snr-fold$fold-ft"

# ckpt_path="nir_fold${fold}_mae"
ckpt_path=$(printf "mae_fold%s_mae" "$fold")

# model="vit"
# ckpt_path=$(printf "nir-snr-fold%s-ft_%s" "$fold" "$model")
# path=$(printf "%s_%s" "$name" "$model")

echo "/mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/mae_rppg/new_exp/$path/ckpt/ckpt.pth"

echo $ckpt_path

apptainer exec /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/apple.sif /bin/bash -c " \
  . /ext3/env.sh; \
  conda activate apple; \
  cd /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/RGB-NIR-rPPG; \
  python ./test.py --dataset ./config/dataset/fold$fold.yaml --dataset_path /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/data/PreprocessedData/MSTMap_new --runner ./config/vit/runner.yaml --model vit --name $name --channels 6 --map_type NIR NIR --ckpt_path /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/mae_rppg/new_exp/$ckpt_path/ckpt/ckpt.pth --task finetune \
"

apptainer exec /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/apple.sif /bin/bash -c " \
  . /ext3/env.sh; \
  conda activate apple; \
  cd /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/RGB-NIR-rPPG; \
  python ./test.py --dataset ./config/dataset/fold$fold.yaml --dataset_path /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/data/PreprocessedData/CHROM_POS --dataset_type full --runner ./config/vit/runner.yaml --model vit --name $name --channels 6 --map_type NIR NIR --ckpt_path /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/mae_rppg/new_exp/$ckpt_path/ckpt/ckpt.pth --task finetune \
"