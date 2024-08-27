import torch

# own modules
from utils_main import N_CLASSES, IMAGE_SIZE
from models.parent_class import BaseModel
from models.fully_convolutional_resnet.fcn_resnet_backbone import ResNet, Bottleneck, BasicBlock
from models.fully_convolutional_resnet.fcn_resnet_head import FCNHead



class FCNResNet(BaseModel):
    
    def __init__(self, block_type = Bottleneck, layers : list = [3, 4, 6, 3], n_classes : int = N_CLASSES):
        
        # Initialize the parent class
        super(FCNResNet, self).__init__()
        
        # Initialize the ResNet Backbone
        self.backbone = ResNet(block_type, layers, n_classes)
        
        # Initialize the FCN Head for the ResNet types
        if block_type == BasicBlock:
            self.classifier = FCNHead(512, n_classes)
            
        elif block_type == Bottleneck:
            self.classifier = FCNHead(2048, n_classes)
        
        else:
            raise ValueError("Invalid block type. Supported types: BasicBlock, Bottleneck")
        
        
    def forward(self, x):
            
        # Forward pass through the backbone
        x = self.backbone(x)
        
        # Forward pass through the head
        x = self.classifier(x)
        
        # Upsample to match the desired output size
        x = torch.nn.functional.interpolate(x, size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), mode='bilinear', align_corners=False)
        
        return x