import torch
from typing import Tuple



class DoubleConv(torch.nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        
        super().__init__()
        
        # two 3x3 convolutions with ReLU activation
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        return self.double_conv(x)
    
    
class DownSampling(torch.nn.Module):
        
    def __init__(self, in_channels: int, out_channels: int):
        
        super().__init__()
        
        # create a DoubleConv layer with the current and next channel size
        self.double_conv = DoubleConv(in_channels, out_channels)
        
        # define the max pooling layer (always 2x2 and stride 2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # apply the double convolution
        x_down = self.double_conv(x)
        
        # apply the max pooling
        x_pol = self.max_pool(x_down)
        
        return x_down, x_pol
    
    
class UpSampling(torch.nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        
        super().__init__()
        
        # create a transposed convolution layer with the current and next channel size
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

        
    def forward(self, x_pol: torch.Tensor, x_down: torch.Tensor) -> torch.Tensor:
        
        # apply the transposed convolution
        x_up = self.up(x_pol)
        
        # concatenate the upsampled tensor with the corresponding tensor from the down path
        x = torch.cat([x_up, x_down], dim=1)
        
        # apply the double convolution and return the result
        return self.conv(x)
