import datetime
import torch
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import get_model
import os
from data.dataloader import RGB_Dataset
import Evaluation.metrics as M
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils.loss import iou_loss
from utils.lr import adjust_learning_rate
import json


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # keep convolution algorithm deterministic
    # torch.backends.cudnn.benchmark = False  # using fixed convolution algorithm to accelerate training
    # if model and input are fixed, set True to search better convolution algorithm
    torch.backends.cudnn.benchmark = True


def train_one_epoch(args, epoch, epochs, model, opt, train_dl, train_size):
    model.train()

    epoch_total_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 = 0

    loss_weights = [1, 1, 1, 1]
    l = 0

    progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch + 1, epochs), ncols=140)
    for i, data_batch in enumerate(progress_bar):
        l = l + 1
        images = data_batch['image']
        label = data_batch['gt']

        H, W = train_size
        images, label = images.cuda(non_blocking=True), label.cuda(non_blocking=True)

        if args.method in ["Tramba-R-SOD", "Tramba-R-TSOD"]:
            mask_1_8, mask_1_4, mask_1_1 = model(images)
            assert mask_1_1.shape[2] == H
            mask_1_8 = F.interpolate(mask_1_8, (H, W), mode='bilinear')
            mask_1_4 = F.interpolate(mask_1_4, (H, W), mode='bilinear')

            loss3 = F.binary_cross_entropy_with_logits(mask_1_8, label) + iou_loss(mask_1_8, label)
            loss2 = F.binary_cross_entropy_with_logits(mask_1_4, label) + iou_loss(mask_1_4, label)
            loss1 = F.binary_cross_entropy_with_logits(mask_1_1, label) + iou_loss(mask_1_1, label)

            loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_total_loss += loss.cpu().data.item()
            epoch_loss1 += loss1.cpu().data.item()
            epoch_loss2 += loss2.cpu().data.item()
            epoch_loss3 += loss3.cpu().data.item()
        else:
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = model(images)
            assert mask_1_1.shape[2] == H
            mask_1_16 = F.interpolate(mask_1_16, (H, W), mode='bilinear')
            mask_1_8 = F.interpolate(mask_1_8, (H, W), mode='bilinear')
            mask_1_4 = F.interpolate(mask_1_4, (H, W), mode='bilinear')

            loss4 = F.binary_cross_entropy_with_logits(mask_1_16, label) + iou_loss(mask_1_16, label)
            loss3 = F.binary_cross_entropy_with_logits(mask_1_8, label) + iou_loss(mask_1_8, label)
            loss2 = F.binary_cross_entropy_with_logits(mask_1_4, label) + iou_loss(mask_1_4, label)
            loss1 = F.binary_cross_entropy_with_logits(mask_1_1, label) + iou_loss(mask_1_1, label)

            loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3 + loss_weights[3] * loss4

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_total_loss += loss.cpu().data.item()
            epoch_loss1 += loss1.cpu().data.item()
            epoch_loss2 += loss2.cpu().data.item()
            epoch_loss3 += loss3.cpu().data.item()
            epoch_loss4 += loss4.cpu().data.item()

        progress_bar.set_postfix(loss=f'{epoch_total_loss / (i + 1):.4f}')

    return epoch_total_loss / l


def test_one_epoch(model, data_root=None, methods='SOD', img_size=384):
    with torch.no_grad():
        model.eval()
        dataset_setname = methods

        FM = M.Fmeasure_and_FNR()
        WFM = M.WeightedFmeasure()
        SM = M.Smeasure()
        EM = M.Emeasure()
        MAE = M.MAE()
        # FNR = M.FNR()

        test_dataset = RGB_Dataset(data_root, ['Test'], img_size, 'Test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
        # progress_bar = tqdm(test_loader, desc=dataset_setname, ncols=140)
        for i, data_batch in enumerate(test_loader):
            images = data_batch['image']
            gt = data_batch['gt'].numpy().squeeze(0).squeeze(0)
            images = Variable(images.cuda())

            outputs_saliency = model(images)
            mask_1_1 = outputs_saliency[-1]
            pred = torch.sigmoid(mask_1_1).data.cpu().numpy().squeeze(0).squeeze(0)
            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            MAE.step(pred=pred, gt=gt)
            # FNR.step(pred=pred, gt=gt)
        fm = FM.get_results()[0]['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']
        fnr = FM.get_results()[1]

        results = {
            'dataset_setname': dataset_setname,
            'Smeasure_r': sm.round(4),
            'Wmeasure_r': wfm.round(4),
            'MAE_r': mae.round(4),
            'adpEm_r': em['adp'].round(4),
            'meanEm_r': em['curve'].mean().round(4),
            'maxEm_r': em['curve'].max().round(4),
            'adpFm_r': fm['adp'].round(4),
            'meanFm_r': fm['curve'].mean().round(4),
            'maxFm_r': fm['curve'].max().round(4),
            'fnr_r': fnr.round(4)
        }
    return results


def record(args, tb_writer, results, epoch, epochs, loss, lr):
    # Record
    os.makedirs(args.save_model, exist_ok=True)
    evaluate_record = os.path.join(args.save_model, "Record_" + args.method + ".txt")
    f = open(evaluate_record, 'a')
    if epoch == 0:
        f.write('\n' + str(datetime.datetime.now()) + '\n')
        f.write('Start record.\n')
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4)
        f.write('Current lr: ' + str(lr) + '\n')

    tb_writer.add_scalar('lr', lr, epoch + 1)
    tb_writer.add_scalar('MAE', results['MAE_r'], epoch + 1)
    tb_writer.add_scalar('adpFm', results['adpFm_r'], epoch + 1)
    tb_writer.add_scalar('meanFm', results['meanFm_r'], epoch + 1)
    tb_writer.add_scalar('maxFm', results['maxFm_r'], epoch + 1)
    tb_writer.add_scalar('adpEm', results['adpEm_r'], epoch + 1)
    tb_writer.add_scalar('meanEm', results['meanEm_r'], epoch + 1)
    tb_writer.add_scalar('maxEm', results['maxEm_r'], epoch + 1)
    tb_writer.add_scalar('Wmeasure', results['Wmeasure_r'], epoch + 1)
    tb_writer.add_scalar('Smeasure', results['Smeasure_r'], epoch + 1)

    eval_record = str(
        'Epoch:' + str(epoch + 1) + '||' +
        'Dataset:' + results['dataset_setname'] + '||' +
        'train_loss' + str(loss) + '; ' +
        'Smeasure:' + str(results['Smeasure_r']) + '; ' +
        'wFmeasure:' + str(results['Wmeasure_r']) + '; ' +
        'MAE:' + str(results['MAE_r']) + '; ' +
        'fnr:' + str(results['fnr_r']) + '||' +
        'adpEm:' + str(results['adpEm_r']) + '; ' +
        'meanEm:' + str(results['meanEm_r']) + '; ' +
        'maxEm:' + str(results['maxEm_r']) + '; ' +
        'adpFm:' + str(results['adpFm_r']) + '; ' +
        'meanFm:' + str(results['meanFm_r']) + '; ' +
        'maxFm:' + str(results['maxFm_r'])
    )
    print(str(
        'MAE:' + str(results['MAE_r']) + '| ' +
        'Smeasure:' + str(results['Smeasure_r']) + '| ' +
        'wFmeasure:' + str(results['Wmeasure_r']) + '| ' +
        'adpEm:' + str(results['adpEm_r']) + '| ' +
        'meanEm:' + str(results['meanEm_r']) + '| ' +
        'maxEm:' + str(results['maxEm_r']) + '| ' +
        'adpFm:' + str(results['adpFm_r']) + '| ' +
        'meanFm:' + str(results['meanFm_r']) + '| ' +
        'maxFm:' + str(results['maxFm_r'])
    ))
    f.write(eval_record)
    f.write("\n")
    print('#' * 50)
    if epoch + 1 == epochs:
        f.write(str(datetime.datetime.now()) + '\n')
        f.write('End Training Record.\n')
    f.close()


def fit(args, model, train_dl, epochs=100, train_size=384, tb_writer=None):
    opt = get_opt(args.lr, model)
    if args.resume is not None:
        if args.resume == "last":
            path = args.save_model + '/' + args.method + '/'
            save_path = path + args.method + "_resume.pth"
            print(save_path)
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'], strict=True)
            opt.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            path = args.resume
            model.load_state_dict(torch.load(path), strict=True)
            filename = os.path.basename(path)
            start_epoch = int(filename.split('_')[-1].split('.')[0])
        print("Model loaded!! Start training from epoch {}!!".format(start_epoch + 1))
    else:
        start_epoch = 0

    decay_epochs = list(map(int, args.decay_epochs.split("-")))
    decay_factors = list(map(float, args.decay_factors.split("-")))
    print(f"decay_epochs: {decay_epochs} || decay_factors: {decay_factors}")
    for epoch in range(start_epoch, epochs):
        # Train
        lr = adjust_learning_rate(opt, epoch, decay_epochs, args.lr, decay_factors)
        loss = train_one_epoch(args, epoch, epochs, model, opt, train_dl, [train_size, train_size])
        # if eval_one_epoch is True or epoch + 1 >= 60:
        if epoch + 1 >= args.see:
            # Evaluation
            results = test_one_epoch(model, data_root=args.evaluation_root,
                                     methods=args.evaluation_dataset,
                                     img_size=args.img_size)
            # Record
            record(args, tb_writer, results, epoch, epochs, loss, lr)
            # Save
            path = args.save_model + '/' + args.method + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            if args.best_MAE is None or results['MAE_r'] < args.best_MAE:
                pattern = path + args.method + "_MAE_*.pth"
                save_path = pattern.replace("*.pth", str(results['MAE_r']) + '_' + str(epoch + 1) + '.pth')
                torch.save(model.state_dict(), save_path)
            if (epoch + 1) % 5 == 0:
                pattern = path + args.method + "_resume.pth"
                save_path = pattern
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, save_path)


def get_opt(lr, model):
    base_params = [params for name, params in model.named_parameters() if ("encoder" in name)]
    # base_names = [name + '\n' for name, params in model.named_parameters() if ("encoder" in name)]
    other_params = [params for name, params in model.named_parameters() if ("encoder" not in name)]
    for name, params in model.named_parameters():
        if "encoder" not in name:
            print(name)
    # other_names = [name + '\n' for name, params in model.named_parameters() if ("encoder" not in name)]
    # 1/10 lr for parameters in backbone
    params = [{'params': base_params, 'lr': lr * 0.1},
              {'params': other_params, 'lr': lr}]
    print(f"Encoder Learning Rate: {lr * 0.1}       " + f"Decoder Learning Rate: {lr}")
    opt = torch.optim.Adam(params, lr)

    return opt


def training(args):
    random_seed(1026)
    tb_writer = SummaryWriter(args.tf_log_path + '/' + args.method)
    print('Starting train..... Model:{}'.format(args.method))
    model = get_model.build(args.method, args)
    train_dataset = RGB_Dataset(root=args.data_root, sets=['Train'], img_size=args.img_size, mode='train')
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=8)

    model.cuda()
    model.train()
    fit(args, model, train_dl, args.train_epochs, args.img_size, tb_writer)
