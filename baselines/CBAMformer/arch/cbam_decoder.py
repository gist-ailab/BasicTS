import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from .attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from torch.nn.modules.transformer import TransformerDecoderLayer

class DecoderBlock(nn.Module):
    def __init__(self, seg_len, d_seg, n_heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(d_seg, n_heads, d_ff, dropout, batch_first=True)
        self.linear_pred = nn.Linear(d_seg, seg_len)

    def forward(self, x, cross):
        dec_out = self.decoder_layer(x, cross) # dec_out = [b, (n_feat n_seg) d_seg]
        
        # dec_out = rearrange(dec_out, 'b, (ts_d seg_dec_num) d_model -> b ts_d seg_dec_num d_model', b=x.shape[0])
        layer_predict = self.linear_pred(dec_out) # [b () n]
        # layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_out, layer_predict




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
        final_predict = None
        i = 0
        x = rearrange(x, 'b n_feat n_seg d_seg -> b (n_feat n_seg) d_seg')
        
        n_feat = x.shape[1]
        for block in self.decode_blocks:
            cross_enc = cross[i]
            x, layer_predict = block(x, cross_enc)
         
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        final_predict = rearrange(final_predict, 'b (n_feat n_seg) seg_len -> b (n_seg seg_len) n_feat', n_feat = n_feat)
        # final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict

