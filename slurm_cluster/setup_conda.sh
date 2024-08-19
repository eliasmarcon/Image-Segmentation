#!/bin/bash

#SBATCH --job-name=conda_setup
#SBATCH -o ./logs/%x-%j.log

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh
conda init bash

# Create a new conda environment
conda create -n cifar python=3.11 -y

# Activate the new environment
conda activate cifar

conda install --yes -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install specific versions of packages from conda-forge
conda install --yes -c conda-forge --file ./requirements.txt

# Deactivate the Conda environment
conda deactivate