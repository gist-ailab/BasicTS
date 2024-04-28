from math import ceil

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .cbam_encoder import Encoder
from .cbam_decoder import Decoder
from .cbam_embed import DSW_embedding
from .cbam import CBAM

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math


# b, ts_len, n_feat, feat_dim
# b, ts_len, n_feat
# b, n_feat, ts_len
# b, n_feat, n_seg, seg_len
# b, n_feat, n_seg, d_seg



class CBAMformerV2(nn.Module):
    def __init__(self, n_feat, in_ts_len, out_ts_len, seg_len, merge_win_size = 4,
                factor=10, d_seg=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False):
        super(CBAMformerV2, self).__init__()
        
        self.n_feat = n_feat
        
        self.in_ts_len = in_ts_len
        self.out_ts_len = out_ts_len
        self.seg_len = seg_len # d_model
    
        self.merge_win_size = merge_win_size
        self.baseline = baseline

        # The padding operation to handle invisible sgemnet length
        self.pad_in_ts_len = ceil(1.0 * in_ts_len / seg_len) * seg_len   # 96
        self.pad_out_ts_len = ceil(1.0 * out_ts_len / seg_len) * seg_len # 336
        self.in_ts_len_add = self.pad_in_ts_len - self.in_ts_len
        
        # Embedding
        # input: B, feat, time_len
        self.d_seg = d_seg
        self.enc_value_embedding = DSW_embedding(seg_len, self.d_seg) # b, n_feat, seg_num, seg_dim
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, n_feat, (self.pad_in_ts_len // seg_len), self.d_seg), requires_grad=True)
        self.pre_norm = nn.LayerNorm(self.d_seg)

        self.cbam = CBAM(n_feat) # b dim seg_num d_model
        # Encoder
        self.encoder = Encoder(e_layers, merge_win_size, d_seg, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_ts_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, n_feat, (self.pad_out_ts_len // seg_len), self.d_seg), requires_grad=True)
        # print('dec_shape: ', (self.pad_out_ts_len // seg_len) * self.seg_len)

        self.decoder = Decoder(seg_len=seg_len, d_layers=e_layers + 1, d_seg=self.d_seg, n_heads=n_heads, d_ff=d_ff, dropout=dropout, \
                                    out_seg_num = (self.pad_out_ts_len // seg_len), factor = factor)

    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        x_seq = history_data[:, :, :, 0]         # b, ts_len, n_feat, (d_feat)
        
        
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0] 
        if (self.in_ts_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_ts_len_add, -1), x_seq), dim = 1)
        
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        x_seq = self.cbam(x_seq) #[b, n_feat, n_seg, d_seg]
        
        enc_out = self.encoder(x_seq)#[b, -1, d_seg]
        dec_in = repeat(self.dec_pos_embedding, 'b n_feat n_seg d_seg -> (repeat b) n_feat n_seg d_seg', repeat = batch_size)
        
        predict_y = self.decoder(dec_in, enc_out)
        predict_y = rearrange(predict_y, 'b n_feat l -> b l n_feat', n_feat=self.n_feat)
        # dec_in = dec_in.reshape(dec_in.shape[0], -1, dec_in.shape[-1] )
        pred = base + predict_y[:, :self.out_ts_len, :] # (batch_size, out_len, data_dim)
        return pred.unsqueeze(-1)



# class CBAMformerRouter(nn.Module):
#     def __init__(self, n_feat, in_ts_len, out_ts_len, seg_len, merge_win_size = 4,
#                 factor=10, d_seg=512, d_ff = 1024, n_heads=8, e_layers=3, 
#                 dropout=0.0, baseline = False):
#         super(CBAMformerRouter, self).__init__()
        
#         self.n_feat = n_feat
        
#         self.in_ts_len = in_ts_len
#         self.out_ts_len = out_ts_len
#         self.seg_len = seg_len # d_model
#         # self.seg_num = ceil(self.in_time_len / self.seg_len)
        
#         self.merge_win_size = merge_win_size
#         self.baseline = baseline

#         # The padding operation to handle invisible sgemnet length
#         self.pad_in_ts_len = ceil(1.0 * in_ts_len / seg_len) * seg_len   # 96
#         self.pad_out_ts_len = ceil(1.0 * out_ts_len / seg_len) * seg_len # 336
#         self.in_ts_len_add = self.pad_in_ts_len - self.in_ts_len
#         # self.n_seg = (self.pad_in_ts_len // seg_len)
        
#         # Embedding
#         # input: B, feat, time_len
#         self.d_seg = d_seg
#         # self.enc_value_embedding = nn.Linear(in_ts_len, self.seg_dim)
#         self.enc_value_embedding = DSW_embedding(seg_len, self.d_seg) # b, n_feat, seg_num, seg_dim

#         self.enc_pos_embedding = nn.Parameter(torch.randn(1, n_feat, (self.pad_in_ts_len // seg_len), self.d_seg), requires_grad=True)
#         self.pre_norm = nn.LayerNorm(self.d_seg)

#         self.cbam = CBAM(n_feat) # b dim seg_num d_model
#         # Encoder
#         self.encoder = EncoderRouter(e_layers, merge_win_size, d_seg, n_heads, d_ff, block_depth = 1, \
#                                     dropout = dropout,in_seg_num = (self.pad_in_ts_len // seg_len), factor = factor)
        
#         # Decoder
#         self.dec_pos_embedding = nn.Parameter(torch.randn(1, n_feat, (self.pad_out_ts_len // seg_len), self.d_seg), requires_grad=True)
#         # print('dec_shape: ', (self.pad_out_ts_len // seg_len) * self.seg_len)

#         self.decoder = DecoderRouter(seg_len=seg_len, d_layers=e_layers + 1, d_seg=self.d_seg, n_heads=n_heads, d_ff=d_ff, dropout=dropout, \
#                                     out_seg_num = (self.pad_out_ts_len // seg_len), factor = factor)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         x_seq = history_data[:, :, :, 0]         # b, ts_len, n_feat, (d_feat)
        
        
#         if (self.baseline):
#             base = x_seq.mean(dim = 1, keepdim = True)
#         else:
#             base = 0
#         batch_size = x_seq.shape[0] 
#         if (self.in_ts_len_add != 0):
#             x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_ts_len_add, -1), x_seq), dim = 1)
        
#         x_seq = self.enc_value_embedding(x_seq)
#         x_seq += self.enc_pos_embedding
#         x_seq = self.pre_norm(x_seq)
#         x_seq = self.cbam(x_seq) #[b, n_feat, n_seg, d_seg]
#         # x_seq = x_seq.reshape(x_seq.shape[0], -1, x_seq.shape[-1]) # [B, seg_num, d_model]
#         enc_out = self.encoder(x_seq)#[b, -1, d_seg]


#         # enc_out = rearrange(enc_out, 'b (n_feat n_seg) d_seg -> b n_feat n_seg d_seg', n_feat=self.n_feat)
#         # dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
#         dec_in = repeat(self.dec_pos_embedding, 'b n_feat n_seg d_seg -> (repeat b) n_feat n_seg d_seg', repeat = batch_size)
#         # dec_in = repeat(self.dec_pos_embedding, 'b n_feat d_seg -> b n_feat d_seg', b=batch_size)
        
#         predict_y = self.decoder(dec_in, enc_out)
#         # dec_in = dec_in.reshape(dec_in.shape[0], -1, dec_in.shape[-1] )
#         pred = base + predict_y[:, :self.out_ts_len, :] # (batch_size, out_len, data_dim)
#         return pred.unsqueeze(-1)