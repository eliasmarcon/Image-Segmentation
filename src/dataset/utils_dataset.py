
import torch
import torchvision.transforms.v2

from pathlib import Path
from typing import Tuple

# Own modules
from dataset.cityscapes import CityscapesCustom





def get_transformations(image_size : tuple = (1024, 2048)): # image_size --> (height, width)    
    
    
    img_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(), 
                                                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                    # torchvision.transforms.v2.Resize(size=(image_size[0], image_size[1]), interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST), # (h, w)
                                                    torchvision.transforms.v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    target_transform = torchvision.transforms.Compose([torchvision.transforms.v2.ToImage(), 
                                                        torchvision.transforms.v2.ToDtype(torch.long, scale=False)])
                                                        # torchvision.transforms.v2.Resize(size=(image_size[0], image_size[1]), interpolation=torchvision.transforms.v2.InterpolationMode.NEAREST)])
                            
    return img_transform, target_transform


def get_cityscapes_datasets(root : Path = "./cityscapes_assg2/", val_split : float = 0.2) -> Tuple[CityscapesCustom, CityscapesCustom, CityscapesCustom]:
    
    
    # Get the transformations
    img_transform, target_transform = get_transformations()
    
    # Load the full training dataset
    full_train_data = CityscapesCustom(
        root=root, 
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
        root=root, 
        split="val",  # Use the original validation set as test set
        mode="fine",
        target_type='semantic', 
        transform=img_transform,
        target_transform=target_transform
    )
    
    return train_data, val_data, test_data


def create_dataloaders(dataset_type : str, dataset : CityscapesCustom, batch_size : int) -> torch.utils.data.DataLoader:
    
    # Create a DataLoader for the given dataset
    if dataset_type == "train":
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers = 4,
                                           persistent_workers=True)
    
    elif dataset_type == "val":
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers = 4,
                                           persistent_workers=True)
    
    elif dataset_type == "test":
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False, 
                                           num_workers = 1)
        