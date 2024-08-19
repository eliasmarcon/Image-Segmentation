#!/bin/bash

#SBATCH --job-name=cifar10
#SBATCH -o ./logs/%x-%j.log
#SBATCH --exclusive


source /opt/conda/etc/profile.d/conda.sh
conda activate cifar


# Assign the hyperparameter file argument
file_number="$1"
PARAM_FILE="./hyperparameter_sets/hyperparameters_${file_number}.txt"


# Check if the parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Parameter file 'hyperparameter_sets/$PARAM_FILE' not found!"
  exit 1
fi

# Read hyperparameters from the file into an array
IFS=$'\n' read -d '' -r -a hyperparameters < "$PARAM_FILE"

# Loop over each set of hyperparameters
for params in "${hyperparameters[@]}"; do
    set -- $params
    num_epochs=$1
    batch_size=$2
    model_type=$3
    scheduler=$4
    learning_rate=$6
    weight_decay=$7
    gamma=$8

    if [ "$5" == "True" ]; then
        AUGMENTATION="--data_augmentation"
    else
        AUGMENTATION=""
    fi

    # Run the training script
    srun python ./src/main.py \
        --num_epochs $num_epochs \
        --batch_size $batch_size \
        --model_type $model_type \
        --scheduler $scheduler \
        $AUGMENTATION \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --gamma $gamma
                              
done

# Deactivate Conda environment
conda deactivate
