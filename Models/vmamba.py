import math
import re
from functools import partial
from typing import Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from Models.SS2D.csms6s import CrossScan, CrossMerge, CrossScan_Line, CrossMerge_Line
from Models.modules import Linear2d, LayerNorm2d, Mlp, Permute
from Models.mamba_init import D_init, Dt_init, A_log_init
from Models.SS2D.csms6s import SelectiveScanOflex

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SS2Dv2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            disable_z=True,
            # ======================
            scan=CrossScan,
            merge=CrossMerge,
            k_group=4,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardv2

        self.disable_z = disable_z
        self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================

        self.forward_core = partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                                    no_einsum=True, CrossScan=scan, CrossMerge=merge)
        k_group = k_group

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                Dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = D_init(d_inner, copies=k_group, merge=True)  # (K * D)

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1),
                             normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous()
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3).contiguous()  # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1).contiguous()
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        z = None
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, SS2Dv2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
            # dwconv ===============
            # < 2 means no conv
            d_conv=3, conv_bias=False,
            # ======================
            dropout=0.0, bias=False,
            # dt init ==============
            dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, initialize="v0",
            # ======================
            forward_type="v2", channel_first=False,
            # ======================
            disable_z=True,
            scan=CrossScan,
            merge=CrossMerge,
            k_group=4,
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, disable_z=disable_z,
            scan=scan, merge=merge, k_group=k_group,
        )
        self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 1,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=False,
            ssm_drop_rate: float = 0.0,
            ssm_init="v0",
            forward_type="v05",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class VSSMEncoder(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            depths=[2, 2, 15, 2],
            dims=128,
            # =========================
            drop_path_rate=0.6,
            patch_norm=True,
            norm_layer="LN2D",  # "BN", "LN2D"
            # =========================
            posembed=False,
            imgsize=224,
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        norm_layer = LayerNorm2d
        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        self.patch_embed = self._make_patch_embed_v2(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                                     channel_first=self.channel_first)

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        _make_downsample = self._make_downsample_v3
        for i_layer in range(self.num_layers):
            self.downsample.append(_make_downsample(
                self.dims[i_layer],
                # self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity())
            self.layers.append(self._make_layer(
                dims=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ))

        self.apply(self._init_weights)

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, (dim * 2) if out_dim < 0 else out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer((dim * 2) if out_dim < 0 else out_dim),
        )

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=128, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_layer(dims=128, drop_path=[0.1, 0.1], norm_layer=nn.LayerNorm, channel_first=False, **kwargs, ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(hidden_dim=dims, drop_path=drop_path[d], norm_layer=norm_layer,
                                   channel_first=channel_first))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            # downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x_ret = []
        x_ret.append(x)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1).contiguous() if not self.channel_first else self.pos_embed
            x = x + pos_embed

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x)  # .permute(0, 3, 1, 2)
            if s < len(self.downsample):
                x = self.downsample[s](x)
        return x_ret


# =====================================================
class VSSMDecoderBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=True,
            # =============================
            ssm_d_state: int = 1,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=False,
            ssm_drop_rate: float = 0.0,
            ssm_init="v0",
            forward_type="v05",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            post_norm: bool = False,
            # =============================
            scan=CrossScan,
            merge=CrossMerge,
            k_group=4,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm1 = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                # ==========================
                disable_z=True,
                scan=scan,
                merge=merge,
                k_group=k_group,
            )
        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


# =====================================================
class DWConv(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                                 bias=bias,
                                 groups=in_channels)

    def forward(self, x):
        return self.dw_conv(x)


class DWMSMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc3 = DWConv(hidden_features, 3, True)
        self.dwc5 = DWConv(hidden_features, 5, True)
        self.dwc7 = DWConv(hidden_features, 7, True)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.dwc3(x) + self.dwc5(x) + self.dwc7(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiScaleDecoderBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=True,
            # =============================
            ssm_d_state: int = 1,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=False,
            ssm_drop_rate: float = 0.0,
            ssm_init="v0",
            forward_type="v05",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            post_norm: bool = False,
            # =============================
            scan=CrossScan_Line,
            merge=CrossMerge_Line,
            k_group=8,
            # scan=CrossScan,
            # merge=CrossMerge,
            # k_group=4,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm1 = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                # ==========================
                disable_z=True,
                scan=scan,
                merge=merge,
                k_group=k_group,
            )
        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = DWMSMlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


def load_pretrained_Base(model, ckpt_path=""):
    print(f"Loading weights from: {ckpt_path}")
    skip_params = []

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()
    init_modules = []
    for k, v in ckpt['model'].items():
        if "classifier" in k:
            print(f"Passing weights: {k}")
            continue
        if "downsample" in k:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", k)[0])
            k = k.replace(f"layers.{i_ds}.downsample", f"downsample.{i_ds}")
            assert k in model_dict.keys()
        if k in model_dict.keys():
            assert v.shape == model_dict[k].shape, f"module: {k} Shape mismatch: {v.shape} vs {model_dict[k].shape}"
            model_dict[k] = v
            init_modules.append(k)
        else:
            print(f"Module can not find: {k}")
    model.load_state_dict(model_dict)
    for k in model_dict.keys():
        if k not in init_modules:
            print(f"Module {k} has not been inited!")
    return model


if __name__ == '__main__':
    x = torch.randn((1, 3, 384, 384)).cuda()
    model = VSSMEncoder(imgsize=384, dims=128).cuda()
    feas = model(x)
    for fea in feas:
        print(fea.shape)
    load_pretrained_Base(model, ckpt_path='/root/autodl-tmp/M3Net/pretrained_model/vssm_base_0229_ckpt_epoch_237.pth')
