import torch
from enum import Enum

# own modules
import utils_main
from models.fully_convolutional_resnet.fcn_resnet import FCNResNet
from models.fully_convolutional_resnet.fcn_resnet_backbone import Bottleneck, BasicBlock
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet



class ModelType(Enum):
    
    RESNET = "resnet"
    SEGFORMER = "segformer"
    UNET = "unet"


def create_model(model_type : ModelType) -> torch.nn.Module:
    
    if ModelType.RESNET.value in model_type:
        
        model_size = str(model_type.split('_')[1])
        
        if model_size == '18':
            
            model = FCNResNet(BasicBlock, [2, 2, 2, 2], utils_main.N_CLASSES)
            
        elif model_size == '34':
            
            model = FCNResNet(BasicBlock, [3, 4, 6, 3], utils_main.N_CLASSES)
            
        elif model_size == '50':
            
            model = FCNResNet(Bottleneck, [3, 4, 6, 3], utils_main.N_CLASSES)
            
        elif model_size == '101':
            
            model = FCNResNet(Bottleneck, [3, 4, 23, 3], utils_main.N_CLASSES)
            
        elif model_size == '152':
            
            model = FCNResNet(Bottleneck, [3, 8, 36, 3], utils_main.N_CLASSES)
            
        else:
            raise ValueError(f"Model configuration for {model_type} not found")
        
        
    elif ModelType.SEGFORMER.value in model_type:
        
        model_size = model_type.split('_')[1]
        
        if model_size == "small":
            
            embed_dims=[32, 64, 160, 256]
            num_heads=[1, 2, 5, 8]
            
        elif model_size == "base":
            
            embed_dims=[64, 128, 256, 320]
            num_heads=[2, 2, 2, 2]
           
        elif model_size == "large":
            
            embed_dims=[128, 256, 320, 384]
            num_heads=[2, 2, 2, 2]
            
        else:
            raise ValueError(f"Model configuration for {model_type} not found") 
        
        model = SegFormer(embed_dims = embed_dims, num_heads = num_heads, num_classes = utils_main.N_CLASSES)

        
    elif ModelType.UNET.value in model_type:
        
        model_size = model_type.split('_')[1]
        
        if model_size == "small":
            
            channel_list = [64, 128, 256, 512]
            
        elif model_size == "base":
            
            channel_list = [64, 128, 256, 512, 1024]
           
        elif model_size == "large":
            
            channel_list = [64, 128, 256, 512, 1024, 2048]
            
        else:
            raise ValueError(f"Model configuration for {model_type} not found")
          
        model = UNet(input_channels=utils_main.IN_CHANNELS, channel_list = channel_list, num_classes=utils_main.N_CLASSES)
    
    
    else:
        raise ValueError(f"Model {model_type} not found")

    return model