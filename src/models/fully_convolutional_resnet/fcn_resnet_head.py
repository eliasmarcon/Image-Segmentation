import torch



class FCNHead(torch.nn.Sequential):
    
    
    def __init__(self, in_channels : int, out_channels : int):
        
        super(FCNHead, self).__init__()
        
        inter_channels = in_channels // 4 # intermediate channels for the head as output
        
        # Convolution Layer
        self.conv = torch.nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False)
        
        # Batch Normalization
        self.bn = torch.nn.BatchNorm2d(inter_channels)
        
        # ReLU Activation
        self.relu = torch.nn.ReLU()
        
        # Dropout 0.1
        self.dropout = torch.nn.Dropout(0.1)
        
        # Convolution Layer for classification
        self.conv_class = torch.nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        
    
    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_class(x)
        
        return x