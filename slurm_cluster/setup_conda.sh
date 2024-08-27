#!/bin/bash

#SBATCH --job-name=conda_setup
#SBATCH -o ./logs/%x-%j.log

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh
conda init bash

# Get the Conda environment name
env_name="image_segmentation"

# Check if the Conda environment already exists
if conda env list | grep -q "$env_name"; then
    echo "Conda environment '$env_name' already exists!"
    exit 1
fi

# Create a new conda environment
conda create -n $env_name python=3.11 -y

# Activate the new environment
conda activate $env_name

# Install PyTorch and other required packages
conda install --yes -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install specific versions of packages from conda-forge
conda install --yes -c conda-forge --file ./requirements.txt

# Deactivate the Conda environment
conda deactivate