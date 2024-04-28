import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from .attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil
from torch import Tensor
from typing import Callable, Optional
from torch.nn.modules.transformer import TransformerEncoderLayer
from .attn import OneStageAttentionLayer

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, seg_dim, win_size, norm_layer=nn.LayerNorm):
        super(SegMerging, self).__init__()

        self.seg_dim = seg_dim
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * seg_dim, seg_dim)
        self.norm = norm_layer(win_size * seg_dim)

    def forward(self, x):
        """
        x: B, n_feat, n_seg, d_model(seg_dim)
        """
        batch_size, n_feat, seg_num, seg_dim = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, n_feat, seg_num/win_size, seg_dim*win_size] 

        x = self.norm(x) # [B, n_feat, seg_num/win_size, seg_dim*win_size] 
        x = self.linear_trans(x) # [B, n_feat, seg_num/win_size, seg_dim]

        return x

class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''
    def __init__(self, win_size, seg_dim, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()
       
        if (win_size > 1):
            self.merge_layer = SegMerging(seg_dim, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TransformerEncoderLayer(seg_dim, n_heads, \
                                                            d_ff, dropout, batch_first=True))
    
    def forward(self, x, n_feat): # b, n_feat, n_seg, d_seg
        
        
        if self.merge_layer is not None: 
            x = rearrange(x, '(b n_feat) n_seg d_seg -> b n_feat n_seg d_seg', n_feat = n_feat)
            x = self.merge_layer(x) 
         
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # b, n_feat n_seg*d_seg
        for layer in self.encode_layers: #0: x =>  
            x = layer(x)       
        
        
        return x

class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, e_blocks, win_size, seg_dim, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, seg_dim, n_heads, d_ff, block_depth, dropout,\
                                            ))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, seg_dim, n_heads, d_ff, block_depth, dropout,\
                                            ))

    def forward(self, x): #[b, n_feat, n_seg, d_seg]
        encode_x = []
        n_feat = x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x, n_feat)
            encode_x.append(x)

        return encode_x
    
