#!/bin/bash

#SBATCH --job-name=multi_seg
#SBATCH -o ./logs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive

source /opt/conda/etc/profile.d/conda.sh
conda activate image_segmentation

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"
# export LOGLEVEL=INFO

# Assign the hyperparameter file argument
folder_type="$1"
file_number="$2"

# Ensure folder_type and file_number are set
if [[ -z "$folder_type" || -z "$file_number" ]]; then
    echo "Error: 'folder_type' and 'file_number' must be set."
    exit 1
fi

# Check if folder_type is one of the valid options and set PARAM_FILE
if [[ "$folder_type" == "first_tests" || "$folder_type" == "second_tests" || "$folder_type" == "third_tests" ]]; then
    PARAM_FILE="./hyperparameter_sets/${folder_type}/hyperparameters_${file_number}.txt"
else
    echo "Error: Invalid folder type. Valid types are 'first_tests', 'second_tests', 'third_tests'."
    exit 1
fi

# Check if PARAM_FILE exists
if [[ ! -f "$PARAM_FILE" ]]; then
    echo "Error: Parameter file '$PARAM_FILE' does not exist."
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

    srun torchrun \
        --nnodes 4 \
        --nproc_per_node 1 \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        --rdzv_endpoint $head_node_ip:29500 \
        ./multi_gpu/main_ddp.py \
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
