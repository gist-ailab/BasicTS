import torch
import torch.nn as nn
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class FeatureAttention(nn.Module): # b, featured_dim, d_model
#     def __init__(self, feature_dim, reduction_ratio=16):
#         self.feature_dim = feature_dim
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.MLP = nn.Sequential(
#             Flatten()
#             nn.Linear(feature_dim, feature_dim // reduction_ratio)
#             nn.ReLU(),
#             nn.Linear(input_channels // reduction_ratio, input_channels)
#         )
#     def forward(self, x): # b, feat_dim, 
#         avg_values = self.avg_pool(x)
#         max_values = self.max_pool(x)
#         out = self.MLP(avg_values) + self.MLP(max_values)
#         scale = x * torch.sigmoid(out)






class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x): # [b, n_feat, n_seg, d_seg]
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x) #[b, n_feat, 1, 1]
        max_values = self.max_pool(x) #[b, n_feat, 1, 1]
        out = self.MLP(avg_values) + self.MLP(max_values) 
        #[b, n_feat] -> [b, n_feat]
        scale = x * torch.sigmoid(out).unsqueeze(1).unsqueeze(2).expand_as(x)
        #[b, n_feat, n_seg, d_seg]
        return scale
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x): # x = [b, n_feat, n_seg, d_seg]
        avg_out = torch.mean(x, dim=1, keepdim=True) # avg_out = [b, 1, n_seg, d_seg]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # max_out = [b, 1, n_seg, d_seg]
        out = torch.cat([avg_out, max_out], dim=1) # out = [b, 2, n_seg, d_seg]
        out = self.conv(out) # [b, 1, a(n_seg), a(d_seg)]
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale  # [b, n_feat, a(n_seg), a(d_seg)]
      
class CBAM(nn.Module): 
    def __init__(self, n_feat, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(n_feat, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x): # batch, n_feat, n_seg, seg_dim
        out = self.channel_att(x) # out = [b, n_feat, n_seg, d_seg]
        out = self.spatial_att(out) 
        return out #[b, n_feat, n_seg, d_seg]

# class CBAM_Timeseries(nn.Module):
#     def __init__(self, feature_dim, reduction_ratio=16, kernel_size=7):
#         super(CBAM_Timeseries, self).__init__()
#         self.feature_att = FeatureAttention(feature_dim, reduction_ratio=reduction_ratio)

