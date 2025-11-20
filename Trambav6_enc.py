from typing import Union, List, Tuple
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from Models.modules import PatchExpand, FinalPatchExpand_X4
from Models.vmamba import VSSMEncoder, load_pretrained_Base, LayerNorm2d, Linear2d, MultiScaleDecoderBlock
from Models.freq_mamba import FreqBlockv6
from Models.SS2D.csms6s import CrossScan, CrossMerge, CrossScan_Line, CrossMerge_Line
from collections import OrderedDict
from Models.encoder.swin_encoder import SwinTransformer
from Models.encoder.resnet_encoder import ResNet
from Models.encoder.pvtv2_encoder import pvt_v2_b4
from torch.utils import model_zoo

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class VSSMDecoder(nn.Module):
    def __init__(
            self,
            deep_supervision,
            features_per_stage: Union[Tuple[int, ...], List[int]] = None,
            drop_path_rate=0.,
            depths=None,
            img_size=384,
            channel_first=True,
    ):
        super().__init__()
        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2] if depths is None else depths

        input_resolution = [img_size // 2 ** len(depths), img_size // 2 ** len(depths)]

        norm_layer = LayerNorm2d
        self.channel_first = channel_first

        self.stage_layers = nn.ModuleList()
        self.expand_layers = nn.ModuleList()
        self.guide_layers = nn.ModuleList()
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
            self.guide_layers.append(FreqBlockv6(
                dim=input_features_skip,
                num_heads=4,
                input_resolution=(input_resolution[0] * (2 ** (stage - 1)),
                                  input_resolution[1] * (2 ** (stage - 1))),
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0., attn_drop=0., drop_path=0.,
                norm_layer=LayerNorm2d,
                local_ws=2,
                alpha=0.5,
                # l_scan=CrossScan, l_merge=CrossMerge, # if debug
                # h_scan=CrossScan, h_merge=CrossMerge, # if debug
                # channel_first=self.channel_first,
            ))
            self.stage_layers.append(self._make_layer(
                dims=input_features_skip,
                drop_path=dpr[sum(depths[:stage - 1]):sum(depths[:stage])],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
                # scan=CrossScan, merge=CrossMerge, # if debug
            ))
            self.seg_layers.append(nn.Conv2d(input_features_skip, 1, 1, 1, 0, bias=True))
            Linear = Linear2d if self.channel_first else nn.Linear
            self.concat_back_dim.append(Linear(input_features_below//2+input_features_skip, input_features_skip))

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
                    scan=CrossScan_Line, merge=CrossMerge_Line, **kwargs, ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(MultiScaleDecoderBlock(hidden_dim=dims, drop_path=drop_path[d], norm_layer=norm_layer,
                                                 channel_first=channel_first, scan=scan, merge=merge))
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
                mid = self.guide_layers[s](skips[-(s + 2)])
                # print(f"stage:{s}  expand_layers input:{lres_input.shape}  output:{x.shape}")
                # print(f"stage:{s}  guide_layers input:{skips[-(s + 2)].shape}  output:{mid.shape}")
                if self.channel_first:
                    x = torch.cat((x, mid), 1)
                else:
                    x = torch.cat((x, mid.permute(0, 2, 3, 1).contiguous()), -1)
                # print(f"stage:{s}  concat output:{x.shape}")
                x = self.concat_back_dim[s](x)
            if self.channel_first:
                x = self.stage_layers[s](x)
            else:
                x = self.stage_layers[s](x).permute(0, 3, 1, 2).contiguous()
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
    def __init__(self, enc_type, decoder_args):
        super().__init__()
        self.enc_type = enc_type
        print(f"Cur Encoder Type: {self.enc_type}!!!")
        if self.enc_type == "Tramba-S-TSOD" or self.enc_type == "Tramba-S-SOD":
            ### Swin Encoder ###
            self.encoder = SwinTransformer(
               img_size=384, 
               embed_dim=128,
               depths=[2,2,18,2],
               num_heads=[4,8,16,32],
               window_size=12
            )
            pretrained_dict = torch.load('/root/autodl-tmp/TSOD/pretrained_model/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)
            
            decoder_args["features_per_stage"] = [128, 256, 512, 1024]
            decoder_args["depths"] = [2, 2, 2, 2]
            self.decoder = VSSMDecoder(**decoder_args)
        elif self.enc_type == "Tramba-P-TSOD" or self.enc_type == "Tramba-P-SOD":
            ### PVT Encoder ###
            self.encoder = pvt_v2_b4()
            pretrained_dict = torch.load('/root/autodl-tmp/TSOD/pretrained_model/pvt_v2_b4.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)
            
            decoder_args["features_per_stage"] = [64, 128, 320, 512]
            decoder_args["depths"] = [2, 2, 2, 2]
            self.decoder = VSSMDecoder(**decoder_args)
        elif self.enc_type == "Tramba-R-TSOD" or self.enc_type == "Tramba-R-SOD":
            ### ResNet Encoder ###    
            self.encoder = ResNet() 
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            encoder_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            encoder_dict.update(pretrained_dict)
            self.encoder.load_state_dict(encoder_dict) 
            
            decoder_args["features_per_stage"] = [256, 512 ,1024]
            decoder_args["depths"] = [2, 2, 2]
            self.decoder = VSSMDecoder(**decoder_args)
        else:
            raise ValueError(f"Unsupported encoder type: {self.enc_type}")

    def forward(self, x):
        skips = [x]
        outs = self.encoder(x)
        if self.enc_type == "Tramba-S-TSOD" or self.enc_type == "Tramba-S-SOD":
            skips += outs[1:][::-1]
        elif self.enc_type == "Tramba-R-TSOD" or self.enc_type == "Tramba-R-SOD":
            skips += outs[1:-1][::-1]
        elif self.enc_type == "Tramba-P-TSOD" or self.enc_type == "Tramba-P-SOD":
            skips += outs[::-1]
        
        out = self.decoder(skips)
        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def bulid_model(
        enc_type,
        deep_supervision: bool = True,
        img_size: int = 384
):
    decoder_args = dict(
        deep_supervision=deep_supervision,
        features_per_stage=None,
        depths=None,
        img_size=img_size,
        drop_path_rate=0.2,
    )

    model = BaseUMamba(enc_type, decoder_args)

    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    model_name = "Tramba-R-TSOD"
    model = bulid_model(
        enc_type=model_name,
        deep_supervision=True,
        img_size=384).cuda()
    input = torch.randn(1, 3, 384, 384).cuda()
    out = model(input)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in {model_name}: {str(total_params / 1000 ** 2)}")
    flops = FlopCountAnalysis(model, input)
    print(f"Total number of parameters in {model_name}: {flops.total() / 1e9}")
