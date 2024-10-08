
import random
import torch
import torchvision.transforms.v2

from pathlib import Path
from typing import Tuple
from enum import Enum

from torch.utils.data.distributed import DistributedSampler

# Own modules
from dataset.cityscapes import CityscapesCustom
from utils_main import IMAGE_SIZE



class DatasetType(Enum):
    
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def get_transformations(augmentation : bool = False):
    
    torch.manual_seed(100)
    random.seed(100)
    
    if augmentation:
        
        img_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(), 
                                                        torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                        torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                                                        torchvision.transforms.v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                                        torchvision.transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                                        torchvision.transforms.v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                                        torchvision.transforms.v2.Resize(size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST), # (h, w)
                                                        torchvision.transforms.v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
        
        target_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(), 
                                                           torchvision.transforms.v2.ToDtype(torch.long, scale=False),
                                                           torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                                                           torchvision.transforms.v2.Resize(size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST)])
                                

    else:
    
        img_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(), 
                                                        torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                        torchvision.transforms.v2.Resize(size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST), # (h, w)
                                                        torchvision.transforms.v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
        
        target_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(), 
                                                           torchvision.transforms.v2.ToDtype(torch.long, scale=False),
                                                           torchvision.transforms.v2.Resize(size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST)])
                                
    return img_transform, target_transform


def get_cityscapes_datasets(base_path : Path = "./cityscapes_data/", augmentation : bool = False, val_split : float = 0.2) -> Tuple[CityscapesCustom, CityscapesCustom, CityscapesCustom]:
    
    # Get the transformations
    img_transform, target_transform = get_transformations(augmentation)
    
    # Load the full training dataset
    full_train_data = CityscapesCustom(
        root=base_path, 
        split="train",
        mode="fine",
        target_type='semantic', 
        transform=img_transform,
        target_transform=target_transform
    )

    # Calculate sizes for train and validation splits
    total_size = len(full_train_data)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Create Generator for random split
    generator = torch.Generator().manual_seed(42)
    # Split the dataset
    train_data, val_data = torch.utils.data.random_split(full_train_data, [train_size, val_size], generator=generator)

    # Create a separate test dataset
    test_data = CityscapesCustom(
        root=base_path, 
        split="val",  # Use the original validation set as test set
        mode="fine",
        target_type='semantic', 
        transform=img_transform,
        target_transform=target_transform
    )
    
    return train_data, val_data, test_data


def create_dataloaders(dataset_type : DatasetType, dataset : CityscapesCustom, batch_size : int) -> torch.utils.data.DataLoader:
    
    # Create a DataLoader for the given dataset
    if dataset_type == DatasetType.TRAIN:
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers = 4,
                                           persistent_workers=True)
    
    elif dataset_type == DatasetType.VAL:
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers = 4,
                                           persistent_workers=True)
    
    elif dataset_type == DatasetType.TEST:
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False, 
                                           num_workers = 1)
        
        
def create_ddp_datalaoders(dataset : CityscapesCustom, batch_size : int) -> torch.utils.data.DataLoader:
        
    return torch.utils.data.DataLoader(
                                        dataset,
                                        batch_size=batch_size,
                                        pin_memory=True,
                                        shuffle=False,
                                        sampler=DistributedSampler(dataset)
                                    )