#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-302 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-24:00:00
#SBATCH -o "/mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/RGB-NIR-rPPG/log/mae_baseline"
# sbatch --array=1-5 ./config/mae/train.sh ./config/mae/train.txt

input_file=$1
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

echo "STARTING..."

echo "-------------------------- fold=$fold --------------------------"


apptainer exec /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/apple.sif /bin/bash -c " \
  . /ext3/env.sh; \
  conda activate apple; \
  cd /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/RGB-NIR-rPPG; \
  python ./test.py --dataset ./config/dataset/fold$fold.yaml --dataset_path /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/data/PreprocessedData/MSTMap_new --runner ./config/mae/runner.yaml --model mae --name mae_fold$fold --channels 6 --map_type NIR --pretrained \
"

apptainer exec /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/apple.sif /bin/bash -c " \
  . /ext3/env.sh; \
  conda activate apple; \
  cd /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/RGB-NIR-rPPG; \
  python ./test.py --dataset ./config/dataset/fold$fold.yaml --dataset_path /mimer/NOBACKUP/groups/naiss2024-23-123/ZiyuanWang/data/PreprocessedData/CHROM_POS --dataset_type full --runner ./config/mae/runner.yaml --model mae --name full_mae_fold$fold --channels 6 --map_type NIR --pretrained \
"