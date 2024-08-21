import utils_main
from models.parent_class import BaseModel
from models.fully_convolutional_resnet.fcn_resnet_backbone import ResNet, Bottleneck
from models.fully_convolutional_resnet.fcn_resnet_head import FCNHead



class FCNResNet(BaseModel):
    
    def __init__(self, block_type = Bottleneck, layers : list = [3, 4, 6, 3], n_classes : int = utils_main.N_CLASSES):
        
        # Initialize the parent class
        super(FCNResNet, self).__init__()
        
        # Initialize the ResNet Backbone
        self.backbone = ResNet(block_type, layers, n_classes)
        
        # Initialize the FCN Head
        self.head = FCNHead(2048, n_classes)
        
        
    def forward(self, x):
            
        # Forward pass through the backbone
        x = self.backbone(x)
        
        print("Backbone output shape: ", x.shape)
        
        # Forward pass through the head
        x = self.head(x)
        
        print("Head output shape: ", x.shape)
        
        return x