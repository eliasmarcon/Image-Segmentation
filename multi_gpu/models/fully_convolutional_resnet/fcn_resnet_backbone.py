import torch
from typing import Type, Union, Optional



# Basic Block for ResNet18 and ResNet34
class BasicBlock(torch.nn.Module):
    expansion: int = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride : int = 1, downsample: Optional[torch.nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        self.downsample = downsample
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace = True)
        
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Bottleneck Block for ResNet50, ResNet101, and ResNet152
class Bottleneck(torch.nn.Module):
    expansion: int = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride : int = 1, downsample: Optional[torch.nn.Module] = None):
        super(Bottleneck, self).__init__()

        # 1x1 convolution
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        
        # 3x3 convolution
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = torch.nn.ReLU(inplace = True)
        self.downsample = downsample
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layer_sizes: list, num_classes: int):
        super(ResNet, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.num_classes = num_classes
        
        self.groups = 1
        self.in_channels = 64
        
        self.normalization_layer = torch.nn.BatchNorm2d
        
        # Initial convolutional layer 1
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_normalization1 = self.normalization_layer(self.in_channels)
        self.relu = torch.nn.ReLU(inplace = True)
        
        # Maxpooling layer
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.residual_block1 = self._make_layer(block, 64, self.layer_sizes[0], stride=1)
        self.residual_block2 = self._make_layer(block, 128, self.layer_sizes[1], stride=2)
        self.residual_block3 = self._make_layer(block, 256, self.layer_sizes[2], stride=2)
        self.residual_block4 = self._make_layer(block, 512, self.layer_sizes[3], stride=2)
        
        
    def _make_layer(self, block, channels: int, num_blocks: int, stride: int = 1):
        downsample = None
    
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.normalization_layer(channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))

        return torch.nn.Sequential(*layers)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Initial convolutional layer 1
        x = self.conv1(x)
        x = self.batch_normalization1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        
        return x