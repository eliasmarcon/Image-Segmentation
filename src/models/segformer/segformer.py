import torch.nn as nn
from functools import partial

import utils_main
from models.parent_class import BaseModel
from models.segformer.mit_transformer import  MixVisionTransformer
from models.segformer.segformer_head import SegFormerHead, resize


# SOURCE: https://github.com/NVlabs/SegFormer

class SegFormer(BaseModel):
    
    def __init__(self, num_classes : int = utils_main.N_CLASSES):
    
        super().__init__()
    
        self.encoder=MixVisionTransformer(
            embed_dims=[32, 64, 160, 256],                 # the feature dimension of each of the 4 blocks
            num_heads=[1, 2, 5, 8],                        # the feature dimension of each of the 4 blocks
            mlp_ratios=[4, 4, 4, 4],                       # the ratio of how much to increase the hidden dim in the MLPs
            qkv_bias=True,                                 # whether to use bias for the q, k, v linear layers
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[2, 2, 2, 2],                           # the depths of each block
            sr_ratios=[8, 4, 2, 1],                        # the ratios used for efficient attention in each block
            drop_rate=0.0, 
            drop_path_rate=0.0)

        self.decoder=SegFormerHead(feature_strides=[4, 8, 16, 32],
                                   in_channels=[32, 64, 160, 256],
                                   in_index=[0,1,2,3],
                                   decoder_params=dict(embed_dim=256),
                                   num_classes=num_classes)

    def forward(self, x):
    
        enc = self.encoder(x)
        out = self.decoder(enc)
        # upsample to the initial resolution
        out = resize(out, size=x.size()[2:], mode='bilinear', align_corners=False)
    
        return out

    