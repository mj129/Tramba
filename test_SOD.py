import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
############
import Evaluation.metrics as M
from torch.autograd import Variable
from data.dataloader import RGB_Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
from get_model import build


def test_one_epoch(args, model, data_root=None, dataset=None):
    with torch.no_grad():
        model.eval()
        test_dataset = RGB_Dataset(data_root, ['Test'], args.img_size, 'Test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)
        save_path = os.path.join(args.image_save_path, args.method, "SOD")
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        progress_bar = tqdm(test_loader, desc=dataset, ncols=140)
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
            cv2.imwrite(os.path.join(save_path, name[0]+'.png'), pred)


def test(args):
    dataset_root_list = {
        # 'DUT-OMRON': '/root/autodl-tmp/SOD_datasets/DUT-OMRON/Test',
        # 'ECSSD': '/root/autodl-tmp/SOD_datasets/ECSSD/Test',
        # 'HKU-IS': '/root/autodl-tmp/SOD_datasets/HKU-IS/Test',
        # 'PASCAL-S': '/root/autodl-tmp/SOD_datasets/PASCAL-S/Test',
        # 'SOD': '/root/autodl-tmp/SOD_datasets/SOD/Test',
        # 'DUTS-Random2500': '/root/autodl-tmp/SOD_datasets/DUTS/Test_Random2500',
        'SOD': '/root/autodl-tmp/SOD_dataset/DUTS',
        }
    model = build(args.method, args)
    model.cuda()
    print(args.resume)
    model.load_state_dict(torch.load(args.resume), strict=True)
    for dataset, dataset_root in dataset_root_list.items():
        print(dataset)
        test_one_epoch(args, model, data_root=dataset_root, dataset=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='Tramba-Ablation-Dil', )
    parser.add_argument('--resume', default=
                        '/root/autodl-tmp/TSOD_results/Tramba-S-SOD/Tramba-S-SOD.pth',
                        type=str, help='checkpoint')
    parser.add_argument('--image_save_path', default='/root/autodl-tmp/TSOD_results', )
    parser.add_argument('--img_size', default=384, )
    parser.add_argument('--pretrained_path', default='/root/autodl-tmp/TEOS/pretrained_model/vssm_base_0229_ckpt_epoch_237.pth', )
    args = parser.parse_args()
    test(args=args)
