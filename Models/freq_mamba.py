import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.vmamba import SS2D
from Models.SS2D.csms6s import CrossScan_Window, CrossMerge_Window, CrossScan_Dilation, CrossMerge_Dilation
from Models.modules import Linear2d, LayerNorm2d, FreqExpand2D, Mlp
from timm.models.layers import DropPath
from Models.DCT_2D import DCT2D


class FreqSS2Dv6(nn.Module):
    def __init__(self, dim, input_resolution=None, norm_layer=LayerNorm2d, 
                 l_scan=CrossScan_Dilation, h_scan=CrossScan_Window,
                 l_merge=CrossMerge_Dilation, h_merge=CrossMerge_Window):
        super().__init__()
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.DCT2D = DCT2D(input_resolution[0], input_resolution[1])
        # Low frequency attention (Lo-Fi)
        self.l_expand = FreqExpand2D(self.dim)
        self.l_ssm = SS2D(
            d_model=self.dim, d_state=1, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
            d_conv=3, conv_bias=False, dropout=0., initialize="v0", channel_first=True, bias=False, disable_z=True,
            scan=l_scan, merge=l_merge, k_group=4,
        )
        # High frequency attention (Hi-Fi)
        self.h_expand = FreqExpand2D(self.dim)
        self.h_ssm = SS2D(
            d_model=self.dim, d_state=1, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
            d_conv=3, conv_bias=False, dropout=0., initialize="v0", channel_first=True, bias=False, disable_z=True,
            scan=h_scan, merge=h_merge, k_group=4,
        )
        # High Low frequency fusion
        self.sig = nn.Sigmoid()
        self.concat_back_dim = Linear2d(self.dim * 2, self.dim, bias=False)

    def hifi(self, x):
        h_out = self.h_ssm(x)
        return h_out

    def lofi(self, x):
        l_out = self.l_ssm(x)
        return l_out

    def forward(self, x):
        high, low = self.DCT2D(x)
        high = self.h_expand(high)
        low = self.l_expand(low)

        hifi_out = self.hifi(high)
        lofi_out = self.lofi(low)

        fusion = torch.cat((hifi_out, lofi_out), 1)
        attn = self.concat_back_dim(fusion)
        x = self.sig(attn) * x
        return x


class FreqBlockv6(nn.Module):
    def __init__(self, dim=128, input_resolution=None, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., 
                 l_scan=CrossScan_Dilation, h_scan=CrossScan_Window,
                 l_merge=CrossMerge_Dilation, h_merge=CrossMerge_Window,
                 act_layer=nn.GELU, norm_layer=LayerNorm2d, local_ws=2, alpha=0.5):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = LayerNorm2d(dim)
        self.attn = FreqSS2Dv6(dim, input_resolution=input_resolution, norm_layer=norm_layer, l_scan=l_scan, h_scan=h_scan, l_merge=l_merge, h_merge=h_merge)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=0.0, channels_first=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
