import torch
import os
from pathlib import Path


class BaseModel(torch.nn.Module):

    def save(self, save_dir: Path, suffix: str = None):
        
        filename = os.path.join(save_dir, (f"{suffix}.pt" if suffix else "model.pt"))
        torch.save(self.state_dict(), filename)


    def load(self, path: Path):
        
        self.load_state_dict(torch.load(path))