from typing import Union, List, Tuple
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from Models.modules import PatchExpand, FinalPatchExpand_X4
from Models.vmamba import VSSMEncoder, load_pretrained_Base, LayerNorm2d, Linear2d, VSSMDecoderBlock
from Models.SS2D.csms6s import CrossScan_Line, CrossMerge_Line, CrossScan_Diagonal, CrossMerge_Diagonal
from Models.SS2D.csms6s import CrossScan_Spiral, CrossMerge_Spiral, CrossScan_Hilbert, CrossMerge_Hilbert
from collections import OrderedDict

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class VSSMDecoder(nn.Module):
    def __init__(
            self,
            deep_supervision,
            features_per_stage: Union[Tuple[int, ...], List[int]] = None,
            drop_path_rate=0.2,
            depths=None,
            channel_first=True,
    ):
        super().__init__()
        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2] if depths is None else depths
        norm_layer = LayerNorm2d
        self.channel_first = channel_first

        self.stage_layers = nn.ModuleList()
        self.expand_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for stage in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-stage]
            input_features_skip = encoder_output_channels[-(stage + 1)]
            self.expand_layers.append(PatchExpand(
                dim=input_features_below,
                dim_scale=2,
                norm_layer=norm_layer,
                channel_first=self.channel_first
            ))
            self.stage_layers.append(self._make_layer(
                dims=input_features_skip,
                drop_path=dpr[sum(depths[:stage - 1]):sum(depths[:stage])],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ))
            self.seg_layers.append(nn.Conv2d(input_features_skip, 1, 1, 1, 0, bias=True))
            Linear = Linear2d if self.channel_first else nn.Linear
            self.concat_back_dim.append(Linear(2 * input_features_skip, input_features_skip))

        # for final prediction
        self.expand_layers.append(FinalPatchExpand_X4(
            dim=encoder_output_channels[0],
            dim_scale=4,
            norm_layer=norm_layer,
            channel_first=self.channel_first,
        ))
        self.stage_layers.append(nn.Identity())
        self.seg_layers.append(nn.Conv2d(input_features_skip, 1, 1, 1, 0, bias=True))

        self.apply(self._init_weights)

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
    def _make_layer(dims=128, drop_path=[0.1, 0.1], norm_layer: nn.Module = LayerNorm2d, channel_first=True,
                    **kwargs, ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSMDecoderBlock(hidden_dim=dims, drop_path=drop_path[d], norm_layer=norm_layer,
                                           channel_first=channel_first, scan=CrossScan_Line, merge=CrossMerge_Line,
                                           k_group=8, ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            # downsample=downsample,
        ))

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stage_layers)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stage_layers) - 1):
                if self.channel_first:
                    x = torch.cat((x, skips[-(s + 2)]), 1)
                else:
                    x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
            if self.channel_first:
                x = self.stage_layers[s](x)
            else:
                x = self.stage_layers[s](x).permute(0, 3, 1, 2)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stage_layers) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


class BaseUMamba(nn.Module):
    def __init__(self, vss_args, decoder_args, use_pretrain=True, pretrained_path=''):
        super().__init__()
        self.vssm_encoder = VSSMEncoder(**vss_args)
        self.decoder = VSSMDecoder(**decoder_args)
        if use_pretrain:
            load_pretrained_Base(self.vssm_encoder,
                                 ckpt_path=pretrained_path)

    def forward(self, x):
        skips = self.vssm_encoder(x)
        out = self.decoder(skips)
        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def get_BaseUMamba(
        deep_supervision: bool = True,
        use_pretrain: bool = True,
        img_size: int = 384,
        dims=128,
        pretrained_path='',
):
    vss_args = dict(
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 15, 2],
        dims=dims,
        # =========================
        drop_path_rate=0.6,
        patch_norm=True,
        norm_layer="LN2D",  # "BN", "LN2D"
        # =========================
        posembed=False,
        imgsize=img_size,
    )

    decoder_args = dict(
        deep_supervision=deep_supervision,
        features_per_stage=[dims, dims * 2, dims * 4, dims * 8],
        drop_path_rate=0.2,
    )

    model = BaseUMamba(vss_args, decoder_args, use_pretrain=use_pretrain, pretrained_path=pretrained_path)

    return model


if __name__ == '__main__':
    from thop import profile
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    model = get_BaseUMamba(use_pretrain=False).cuda()
    input = torch.randn(1, 3, 384, 384).cuda()
    # _ = model(input)
    # from thop import profile
    # flops, params = profile(model, inputs=(input,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {str(total_params / 1000 ** 2)}")
    #
    # print(flops_selective_scan_fn(B=1, L=96*96*4, D=256, N=1, with_Z=False, with_D=True)*2 +
    #       flops_selective_scan_fn(B=1, L=48*48*4, D=512, N=1, with_Z=False, with_D=True)*2 +
    #       flops_selective_scan_fn(B=1, L=24*24*4, D=1024, N=1, with_Z=False, with_D=True)*15 +
    #       flops_selective_scan_fn(B=1, L=12*12*4, D=2048, N=1, with_Z=False, with_D=True)*2 +
    #       flops_selective_scan_fn(B=1, L=96 * 96 * 8, D=256, N=1, with_Z=False, with_D=True) * 2 +
    #       flops_selective_scan_fn(B=1, L=48 * 48 * 8, D=512, N=1, with_Z=False, with_D=True) * 2 +
    #       flops_selective_scan_fn(B=1, L=24 * 24 * 8, D=1024, N=1, with_Z=False, with_D=True) * 2
    #       )
    flops = FlopCountAnalysis(model, input)
    print(flops.total() / 1e9)
    # for out in outs:
    #     print(out.shape)
    # for name, params in model.named_parameters():
    #     print(name)
