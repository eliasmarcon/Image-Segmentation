import torch
import os
from pathlib import Path

import warnings


# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


class BaseModel(torch.nn.Module):

    def save(self, save_dir: Path, suffix: str = None):
        
        filename = os.path.join(save_dir, (f"{suffix}.pt" if suffix else "model.pt"))
        torch.save(self.state_dict(), filename)


    def load(self, path: Path, map_location : str):
        
        self.load_state_dict(torch.load(path, map_location=map_location))