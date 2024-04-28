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
        batch_size = x.shape[0]
        n_feat = x.shape[1]
        final_predict = None
        i = 0 # x= [b, n_feat, n_seg, d_seg]
        x = rearrange(x, 'b n_feat n_seg d_seg -> (b n_feat) n_seg d_seg')
        
        
        for block in self.decode_blocks:
            cross_enc = cross[i]
            x, layer_predict = block(x, cross_enc) # [b, (n_feat n_seg) d_seg], [b (n_feat n_seg), seg_len]
         
            if final_predict is None:
                final_predict = layer_predict 
            else:
                final_predict = final_predict + layer_predict
            i += 1
       
        final_predict = rearrange(final_predict, '(b n_feat) n_seg seg_len -> b n_feat n_seg seg_len', n_feat = n_feat)
        final_predict = final_predict.reshape(final_predict.shape[0], final_predict.shape[1] , -1)
        return final_predict

