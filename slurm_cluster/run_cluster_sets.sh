#!/bin/bash

#SBATCH --job-name=image_seg
#SBATCH -o ./logs/%x-%j.log
#SBATCH --exclusive


source /opt/conda/etc/profile.d/conda.sh
conda activate image_segmentation


# Assign the hyperparameter file argument
folder_type="$1"
file_number="$2"
PARAM_FILE="./hyperparameter_sets/${folder_type}/hyperparameters_${file_number}.txt"


# Check if the parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Parameter file $PARAM_FILE not found!"
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
                              
done

# Deactivate Conda environment
conda deactivate
