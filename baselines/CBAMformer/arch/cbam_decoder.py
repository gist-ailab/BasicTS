import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from .attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from torch.nn.modules.transformer import TransformerDecoderLayer
from .attn import OneStageAttentionLayer, AttentionLayer






class DecoderBlock(nn.Module):
    def __init__(self, seg_len, d_seg, n_heads, d_ff, dropout, out_seg_num = 10, factor = 10):
        super(DecoderBlock, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(d_seg, n_heads, d_ff, dropout, batch_first=True)
        self.linear_pred = nn.Linear(d_seg, seg_len)

    def forward(self, x, cross):
        dec_out = self.decoder_layer(x, cross) 
        ### dec_out = [batch, n_feat, n_seg, d_seg]
       
        layer_predict = self.linear_pred(dec_out) # [b n_feat, n_seg, seg_len]
        # layer_predict = rearrange(layer_predict, 'b n_feat n_seg seg_len -> b (n_feat n_seg) seg_len')

        return dec_out, layer_predict # [b, (n_feat n_seg), d_seg], [b, (n_feat, n_seg), seg_len]

class DecoderBlockRouter(nn.Module):
    def __init__(self, seg_len, d_seg, n_heads, d_ff, dropout, out_seg_num = 10, factor = 10):
        super(DecoderBlockRouter, self).__init__()
        self.self_attn = OneStageAttentionLayer(out_seg_num, factor, d_seg, n_heads, d_ff, dropout)
        self.cross_attn = AttentionLayer(d_seg, n_heads, dropout = dropout)
        
        
        self.norm1 = nn.LayerNorm(d_seg)
        self.norm2 = nn.LayerNorm(d_seg)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_seg, d_seg),
                                nn.GELU(),
                                nn.Linear(d_seg, d_seg))       
        self.linear_pred = nn.Linear(d_seg, seg_len)
    
    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attn(x)
        tmp = self.cross_attn(x, cross, cross)

        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_out = self.norm2(x+y)
        # dec_out = self.decoder_layer(x, cross) 
        ### dec_out = [batch, n_feat, n_seg, d_seg]
       
        layer_predict = self.linear_pred(dec_out) # [b n_feat, n_seg, seg_len]
        # layer_predict = rearrange(layer_predict, 'b n_feat n_seg seg_len -> b (n_feat n_seg) seg_len')

        return dec_out, layer_predict # [b, (n_feat n_seg), d_seg], [b, (n_feat, n_seg), seg_len]

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_seg, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_blocks = nn.ModuleList()
        for i in range(d_layers):
            self.decode_blocks.append(DecoderBlock(seg_len, d_seg, n_heads, d_ff, dropout
                                        ))
    

    def forward(self, x, cross): # b, n_feat, n_seg, d_seg / b, (), d_seg
        n_feat = x.shape[1]
        final_predict = None
        i = 0 # x= [b, n_feat, n_seg, d_seg]
        x = rearrange(x, 'b n_feat n_seg d_seg -> b (n_feat n_seg) d_seg')
        
        
        for block in self.decode_blocks:
            cross_enc = cross[i]
            x, layer_predict = block(x, cross_enc) # [b, (n_feat n_seg) d_seg], [b (n_feat n_seg), seg_len]
         
            if final_predict is None:
                final_predict = layer_predict 
            else:
                final_predict = final_predict + layer_predict
            i += 1
       
        final_predict = rearrange(final_predict, 'b (n_feat n_seg) seg_len -> b n_seg seg_len n_feat', n_feat = n_feat)
        final_predict = final_predict.reshape(final_predict.shape[0], -1 , final_predict.shape[-1])
        # final_predict = rearrange(final_predict, 'b n_feat n_seg seg_len -> b (n_seg seg_len) n_feat', n_feat = n_feat)
       
        # final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict

class DecoderRouter(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_seg, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(DecoderRouter, self).__init__()

        self.router = router
        self.decode_blocks = nn.ModuleList()
        for i in range(d_layers):
            self.decode_blocks.append(DecoderBlockRouter(seg_len, d_seg, n_heads, d_ff, dropout
                                        ))
    

    def forward(self, x, cross): # b, n_feat, n_seg, d_seg / b, (), d_seg
        n_feat = x.shape[1]
        final_predict = None
        i = 0 # x= [b, n_feat, n_seg, d_seg]
        x = rearrange(x, 'b n_feat n_seg d_seg -> b (n_feat n_seg) d_seg')
        
        
        for block in self.decode_blocks:
            cross_enc = cross[i]
            x, layer_predict = block(x, cross_enc) # [b, (n_feat n_seg) d_seg], [b (n_feat n_seg), seg_len]
         
            if final_predict is None:
                final_predict = layer_predict 
            else:
                final_predict = final_predict + layer_predict
            i += 1
       
        final_predict = rearrange(final_predict, 'b (n_feat n_seg) seg_len -> b n_seg seg_len n_feat', n_feat = n_feat)
        final_predict = final_predict.reshape(final_predict.shape[0], -1 , final_predict.shape[-1])
        # final_predict = rearrange(final_predict, 'b n_feat n_seg seg_len -> b (n_seg seg_len) n_feat', n_feat = n_feat)
       
        # final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict
