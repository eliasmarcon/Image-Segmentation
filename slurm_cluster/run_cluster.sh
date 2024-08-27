#!/bin/bash

#SBATCH --job-name=image_seg
#SBATCH -o ./logs/%x-%j.log
#SBATCH --exclusive


source /opt/conda/etc/profile.d/conda.sh
conda activate image_segmentation

# Define variables for positional arguments
num_epochs=$1
batch_size=$2
model_type=$3
learning_rate=$5
weight_decay=$6
gamma=$7

if [ "$4" == "True" ]; then
    AUGMENTATION="--augmentation"
else
    AUGMENTATION=""
fi

# Run the training script
srun python ./src/main.py \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --model_type $model_type \
    $AUGMENTATION \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --gamma $gamma

# Deactivate Conda environment
conda deactivate