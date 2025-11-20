import sys
import torch
from tqdm import tqdm
import Evaluation.metrics as M
from torch.functional import F
import numpy as np
import os
import cv2
# from py_sod_metrics import Fmeasure, Emeasure, Smeasure, MAE, WeightedFmeasure
from torch.autograd import Variable
from data.dataloader import RGB_Dataset
from torch.utils.tensorboard import SummaryWriter
from get_model import build
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', default=384, )
parser.add_argument('--method', default=
                    'Tramba-S',
                   )
parser.add_argument('--pretrained_path', default='/root/autodl-tmp/TSOD/pretrained_model/vssm_base_0229_ckpt_epoch_237.pth', )
parser.add_argument('--data_root', default='/root/autodl-tmp/TSOD_dataset/TSOD10K/', type=str, help='data path')
args = parser.parse_args()

Tramba_list = [
    '/root/autodl-tmp/TSOD_results/Tramba-S/Tramba-S.pth',
]

save_name = args.method
save_root = '/root/autodl-tmp/TSOD_results'
for i in range(len(Tramba_list)):
    path = Tramba_list[i]
    # index = index_list[i]
    print(path)
    model = build(args.method, args)
    model.cuda()
    model.load_state_dict(torch.load(path), strict=True)
    # save_path = os.path.join(save_root, f"Tramba-TSOD-{index}", 'TSOD')
    save_path = os.path.join(save_root, save_name, 'TSOD')
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # data = torch.randn((1, 3, 384, 384)).cuda()
    # measure_inference_speed(model, (data,))
    with torch.no_grad():
        model.eval()
        test_dataset = RGB_Dataset(args.data_root, ['Test'], args.img_size, 'Test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=8)
        progress_bar = tqdm(test_loader, desc='TSOD_Test', ncols=140)
        for i, data_batch in enumerate(progress_bar):
            images = data_batch['image']
            gt = data_batch['gt'].numpy().squeeze(0).squeeze(0)
            shape = data_batch['shape']
            name = data_batch['name']
            images = Variable(images.cuda())
            ############
            outputs_saliency = model(images)
            res = outputs_saliency[-1]
            pred = F.interpolate(res, size=(shape[1], shape[0]), mode='bilinear', align_corners=False)
            pred = torch.sigmoid(pred).data.cpu().numpy().squeeze(0).squeeze(0)
            pred = np.array(pred * 255).astype(np.uint8)
            assert np.min(pred) == 0
            img_save_path = os.path.join(save_path, name[0] + '.png')
            cv2.imwrite(img_save_path, pred)


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps
