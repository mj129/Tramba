
def build(model_name, args):
    model = None
    if model_name == 'BaseUMamba-SOD': # 基线
        from BaseUMamba import get_BaseUMamba as BaseUMamba
        model = BaseUMamba(deep_supervision=True,
                           use_pretrain=True,
                           img_size=args.img_size,
                           dims=128,
                           pretrained_path=args.pretrained_path,
                           )
    elif model_name in ['Tramba-V-TSOD', 'Tramba-V-SOD']: # 最终版本
        from Trambav6 import bulid_model
        model = bulid_model(deep_supervision=True,
                            use_pretrain=True,
                            img_size=args.img_size,
                            dims=128,
                            depths=[2, 2, 2, 2],
                            pretrained_path=args.pretrained_path,
                            )
    elif model_name in ['Tramba-S-TSOD', 'Tramba-P-TSOD', 'Tramba-R-TSOD'] \
      or model_name in ['Tramba-S-SOD',  'Tramba-P-SOD',  'Tramba-R-SOD']: # 其他backbone
        from Trambav6_enc import bulid_model
        ### Swin/PVTv2/ResNet Encoder ###
        model = bulid_model(
            enc_type=model_name,
            deep_supervision=True,
            img_size=args.img_size,
        )

    return model
