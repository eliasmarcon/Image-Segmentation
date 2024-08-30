import torch
from collections import OrderedDict
from typing import List


# own modules
from models.parent_class import BaseModel
from models.unet.utils_unet import DoubleConv, DownSampling, UpSampling



class UNet(BaseModel):
    
    def __init__(self, input_channels: int = 3, channel_list: List[int] = [64, 128, 256, 512], num_classes: int = 19):
    
        super().__init__()
        
        # Append the input channel to the channel list
        channel_list.insert(0, input_channels)       

        # Create the encoder path
        self.encoder_block = OrderedDict[str, DownSampling]()
        
        for i in range(len(channel_list) - 1):
            self.encoder_block[f"encoder_{i + 1}"] = DownSampling(channel_list[i], channel_list[i+1])
        
        self.encoder_block = torch.nn.Sequential(self.encoder_block)

        # Pop input channel from the channel list as it is not needed anymore
        channel_list.pop(0)


        # Create the bottleneck path
        self.bottleneck = DoubleConv(channel_list[-1], channel_list[-1] * 2)
        

        # Create the decoder path
        self.decoder_block = OrderedDict[str, UpSampling]()

        for i in range(len(channel_list) - 1, -1, -1):
            self.decoder_block[f"decoder_{len(channel_list) - i}"] = UpSampling(channel_list[i] * 2, channel_list[i])

        self.decoder_block = torch.nn.Sequential(self.decoder_block)


        # Create the output layer
        self.out = torch.nn.Conv2d(channel_list[0], num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        encoder_outputs: List[torch.Tensor] = []

        # Encoder
        for layer in self.encoder_block:
            x_down, x = layer(x)
            encoder_outputs.append(x_down)


        # Bottleneck
        x = self.bottleneck(x)


        # Decoder
        for i, layer in enumerate(self.decoder_block, start=1):
            x = layer(x, encoder_outputs[-i])

        # Output
        x = self.out(x)
        return x
