import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from .attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from torch.nn.modules.transformer import TransformerDecoderLayer



class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(TransformerDecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict

