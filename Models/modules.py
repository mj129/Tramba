import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Models.swin import window_partition, window_reverse, WindowAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first \
            if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args).contiguous()


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, N):
        return N * (self.in_features + self.out_features) * self.hidden_features


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain: int = 1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, dim, dim_scale=2, norm_layer=LayerNorm2d, channel_first=True):
        super().__init__()
        self.dim = dim
        self.channel_first = channel_first
        Linear = Linear2d if self.channel_first else nn.Linear
        self.expand = Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def _forward_channel_last(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        return x

    def _forward_channel_first(self, x):
        x = self.expand(x)
        B1, C1, H1, W1 = x.shape
        assert H1 == W1
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C1 // 4)
        B2, C2, H2, W2 = x.shape
        assert C2 == C1 // 4 and H2 == 2 * H1 and W2 == 2 * W1
        x = self.norm(x)

        return x

    def forward(self, x):
        return self._forward_channel_first(x) if self.channel_first else self._forward_channel_last(x)


class FinalPatchExpand_X4(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, dim, dim_scale=4, norm_layer=LayerNorm2d, channel_first=True):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.channel_first = channel_first
        self.dim_scale = dim_scale
        Linear = Linear2d if self.channel_first else nn.Linear
        self.expand = Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def _final_patch_expand_channel_first(self, x):
        """
        x: B, C, H*W
        """
        x = self.expand(x)
        B, C, H, W = x.shape
        assert H == W
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x

    def _final_patch_expand_channel_last(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1).contiguous()  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x  # .permute(0, 3, 1, 2)

    def forward(self, x):
        return self._final_patch_expand_channel_first(x) if self.channel_first \
            else self._final_patch_expand_channel_last(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        flops = 0
        # q
        flops += N * self.dim * self.dim * 3
        # qk
        flops += self.num_heads * N * self.dim // self.num_heads * N
        # att v
        flops += self.num_heads * N * self.dim // self.num_heads * N
        # proj
        flops += N * self.dim * self.dim
        return flops


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.Identity):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.dim1 = dim1
        self.dim2 = dim2
        # self.norm1 = norm_layer(dim1)
        # self.norm2 = norm_layer(dim2)
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim1, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim1)

        self.k2 = nn.Linear(dim2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim2, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, fea, depth_fea):
        # fea = self.norm1(fea)
        # depth_fea = self.norm2(depth_fea)
        _, N1, _ = fea.shape
        B, N, _ = depth_fea.shape
        C = self.dim
        q1 = self.q1(fea).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q [B, nhead, N, C//nhead]
        k2 = self.k2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        v2 = self.v2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q1 @ k2.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        fea = (attn @ v2).transpose(1, 2).contiguous().reshape(B, N1, C)
        fea = self.proj(fea)
        fea = self.proj_drop(fea)

        return fea

    def flops(self, N1, N2):
        flops = 0
        # q
        flops += N1 * self.dim1 * self.dim
        # kv
        flops += N2 * self.dim2 * self.dim * 2
        # qk
        flops += self.num_heads * N1 * self.dim // self.num_heads * N2
        # att v
        flops += self.num_heads * N1 * self.dim // self.num_heads * N2
        # proj
        flops += N1 * self.dim * self.dim1
        return flops


class Block(nn.Module):
    # Remove FFN
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self, N):
        flops = 0
        # att
        flops += self.attn.flops(N)
        # norm
        flops += self.dim * N
        return flops


class WindowAttentionBlock(nn.Module):
    r""" Based on Swin Transformer Block, We remove FFN. 
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = self.drop_path(x)

        # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class MixedAttentionBlock(nn.Module):
    def __init__(self, dim, img_size, window_size, num_heads=1, mlp_ratio=3, drop_path=0.):
        super(MixedAttentionBlock, self).__init__()

        self.img_size = img_size
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.windowatt = WindowAttentionBlock(dim=dim, input_resolution=img_size, num_heads=num_heads,
                                              window_size=window_size, shift_size=0,
                                              mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                              drop_path=0.,
                                              act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                              fused_window_process=False)
        self.globalatt = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        att1 = self.windowatt(x)
        att2 = self.globalatt(x)
        x = x + att1 + att2
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x

    def flops(self):
        N = self.img_size[0] * self.img_size[1]
        flops = 0
        flops += self.windowatt.flops()
        flops += self.globalatt.flops(N)
        flops += self.dim * N
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
        return flops


class ConvBNGeLU(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super(ConvBNGeLU, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class FreqMergingExpand2Dv1(nn.Module):
    def __init__(self, dim=96, norm_layer=LayerNorm2d):
        super().__init__()
        self.dim = dim
        Linear = Linear2d
        self.mode = "merge" if dim != 96 else "expand"
        self.out_dim = 2 * self.dim if dim != 96 else 128

        self.h_norm = norm_layer(self.dim)
        self.l_norm = norm_layer(self.dim)
        if self.mode == "merge":
            self.h_reduction = Linear(4 * self.dim, self.out_dim, bias=False)
            self.l_reduction = Linear(4 * self.dim, self.out_dim, bias=False)
        elif self.mode == "expand":
            self.h_expand = Linear(self.dim, self.out_dim * 4, bias=False)
            self.l_expand = Linear(self.dim, self.out_dim * 4, bias=False)
        self.h_act = nn.GELU()
        self.l_act = nn.GELU()

    @staticmethod
    def _patch_merging_channel_first(x):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_expanding_channel_first(x):
        B1, C1, H1, W1 = x.shape
        assert H1 == W1
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C1 // 4)
        B2, C2, H2, W2 = x.shape
        assert C2 == C1 // 4 and H2 == 2 * H1 and W2 == 2 * W1
        return x

    def forward(self, high, low):
        high = self.h_norm(high)
        low = self.l_norm(low)
        if self.mode == "expand":
            high = self.h_expand(high)
            low = self.l_expand(low)
            high = self._patch_expanding_channel_first(high)
            low = self._patch_expanding_channel_first(low)
        else:
            high = self._patch_merging_channel_first(high)
            low = self._patch_merging_channel_first(low)
            high = self.h_reduction(high)
            low = self.l_reduction(low)
        high = self.h_act(high)
        low = self.l_act(low)
        return high, low


class FreqMergingExpand2Dv2(nn.Module):
    def __init__(self, out_dim=128, input_resolution=None):
        super().__init__()
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.high_reconv = ConvBNGeLU(in_channels=96, out_channels=self.out_dim, kernel_size=1)
        self.low_reconv = ConvBNGeLU(in_channels=96, out_channels=self.out_dim, kernel_size=1)

    def forward(self, high, low):
        high = F.interpolate(high, size=self.input_resolution, mode='bilinear', align_corners=False)
        low = F.interpolate(low, size=self.input_resolution, mode='bilinear', align_corners=False)
        high = self.high_reconv(high)
        low = self.low_reconv(low)
        return high, low


class FreqExpand1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.h_expand = nn.Linear(dim, 2 * dim, bias=False)
        self.h_norm = nn.LayerNorm(dim)
        self.l_expand = nn.Linear(dim, 2 * dim, bias=False)
        self.l_norm = nn.LayerNorm(dim)

    def forward(self, high, low):
        high = self.h_expand(high)
        low = self.l_expand(low)
        B, N, C = high.shape
        B, N, C = low.shape

        high = rearrange(high, 'b n (p1 c)-> b (n p1) c', p1=2, c=C // 2)
        low = rearrange(low, 'b n (p1 c)-> b (n p1) c', p1=2, c=C // 2)
        high = self.h_norm(high)
        low = self.l_norm(low)
        return high, low


class FreqExpand2D(nn.Module):
    def __init__(self, dim, norm_layer=LayerNorm2d, channel_first=True):
        super().__init__()
        self.dim = dim
        self.channel_first = channel_first
        Linear = Linear2d if self.channel_first else nn.Linear
        self.expand = Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.expand(x)
        B1, C1, H1, W1 = x.shape
        assert H1 == W1
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C1 // 4)
        B2, C2, H2, W2 = x.shape
        assert C2 == C1 // 4 and H2 == 2 * H1 and W2 == 2 * W1
        x = self.norm(x)

        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
