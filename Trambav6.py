from typing import Union, List, Tuple
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from Models.modules import PatchExpand, FinalPatchExpand_X4
from Models.vmamba import VSSMEncoder, load_pretrained_Base, LayerNorm2d, Linear2d, MultiScaleDecoderBlock
from Models.freq_mamba import FreqBlockv6
from collections import OrderedDict

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


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
                # channel_first=self.channel_first,
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
            blocks.append(MultiScaleDecoderBlock(hidden_dim=dims, drop_path=drop_path[d], norm_layer=norm_layer,
                                                 channel_first=channel_first))
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
                if self.channel_first:
                    x = torch.cat((x, mid), 1)
                else:
                    x = torch.cat((x, mid.permute(0, 2, 3, 1).contiguous()), -1)
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


def bulid_model(
        deep_supervision: bool = True,
        use_pretrain: bool = True,
        img_size: int = 384,
        dims=128,
        depths=[2, 2, 2, 2],
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
        depths=depths,
        img_size=img_size,
        drop_path_rate=0.2,
    )

    model = BaseUMamba(vss_args, decoder_args, use_pretrain=use_pretrain, pretrained_path=pretrained_path)

    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    model = bulid_model(deep_supervision=True,
                        use_pretrain=False,
                        img_size=384,
                        dims=128,
                        depths=[2, 2, 2, 2]
                        ).cuda()
    model.eval()
    """
    torch.Size([4, 3, 384, 384])
    torch.Size([4, 128, 96, 96])
    torch.Size([4, 256, 48, 48])
    torch.Size([4, 512, 24, 24])
    torch.Size([4, 1024, 12, 12])
    """
    high_res_input_size = (1, 3, 384, 384)
    dummy_input = torch.randn(high_res_input_size).cuda()
    
    # --- 4. GPU预热 (Warm-up) ---
    # 第一次运行CUDA操作通常会比较慢，因为需要初始化。
    # 先运行几次，让GPU“热身”，确保计时准确。
    print("Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # --- 5. 精准计时与循环测试 ---
    print("Starting timing...")
    # 使用torch.cuda.Event来获得最精确的GPU执行时间
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500 # 可以设为100-1000
    timings = torch.zeros((repetitions,)).cuda()

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # 等待GPU任务完成
            torch.cuda.synchronize()
            # 记录时间
            curr_time = starter.elapsed_time(ender) # 单位是毫秒 ms
            timings[rep] = curr_time

    mean_syn = torch.mean(timings).item()
    std_syn = torch.std(timings).item()
    fps = 1000.0 / mean_syn

    print(f"Input shape: {high_res_input_size}")
    print(f"Average latency: {mean_syn:.3f} ms")
    print(f"Standard deviation: {std_syn:.3f} ms")
    print(f"Inference speed: {fps:.2f} FPS")
    # input = torch.randn(1, 3, 384, 384).cuda()
    # out = model(input)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters in the model: {str(total_params / 1000 ** 2)}")
    # flops = FlopCountAnalysis(model, input)
    # print(flops.total() / 1e9)
