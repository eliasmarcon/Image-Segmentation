import os
from pathlib import Path


# DEFINE CONSTANTS
IN_CHANNELS = 3
IMAGE_SIZE = (256, 256) # original (1024, 2048) --> (height, width)    
N_CLASSES = 19
TEST_CHECKPOINTS = ['terminal.pt', 'best_val_loss.pt', 'best_val_mIoU.pt']



def create_save_dir(runs_path : str, model_run_name : str) -> Path:
    
    # Create the experiment root folder
    _create_folder(runs_path)
    
    # Create model type folder
    model_type = model_run_name.split("_")[0]
    _create_folder(os.path.join(runs_path, model_type))
    
    # Create the experiment folder
    experiment_root = os.path.join(runs_path, model_type, model_run_name)
    _create_folder(experiment_root)
    
    return experiment_root


def _create_folder(path : str) -> None:

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)