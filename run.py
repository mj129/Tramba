from train import training
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33115', type=str, help='init_method')
    parser.add_argument('--parallel', action='store_true', help="multi-gpu or single GPU rraining")
    
    parser.add_argument('--data_root', default='/root/autodl-tmp/TSOD_dataset/TSOD10K/', type=str, help='data path')
    # parser.add_argument('--data_root', default='/root/autodl-tmp/SOD_dataset/DUTS/', type=str, help='data path')
    parser.add_argument('--train_dataset', default='', type=str, help='dataset for training')
    parser.add_argument('--evaluation_root', default='/root/autodl-tmp/TSOD_dataset/TSOD10K/', type=str)
    # parser.add_argument('--evaluation_root', default='/root/autodl-tmp/SOD_dataset/DUTS/', type=str)
    parser.add_argument('--evaluation_dataset', default='', type=str, help='dataset for evaluation')
    
    parser.add_argument('--img_size', default=384, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/', type=str, help='load Pretrained model')
    # parser.add_argument('--weight_decay', default=1e-4, type=int, help='weight decay for optimizer')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--save_model', default='/root/autodl-tmp/TSOD_results', type=str, help='save model path')
    parser.add_argument('--tf_log_path', default='/root/tf-logs', type=str, help='tf_log_path')
    parser.add_argument('--pretrained_path',
                        default='/root/autodl-tmp/TSOD/pretrained_model/vssm_base_0229_ckpt_epoch_237.pth',
                        type=str, help='pretrained_path')
    parser.add_argument('--resume', default=None, type=str, help='checkpoint')
    
    parser.add_argument('--see', default=40, type=int)
    parser.add_argument('--train_epochs', default=80, type=int, help='total training epochs')
    parser.add_argument('--decay_epochs', default="60", type=str)
    parser.add_argument('--decay_factors', default="0.2", type=str)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for optimizer')
    parser.add_argument('--method', default=None, type=str, help='different backbone')
    parser.add_argument('--best_MAE', default=None, type=float)

    args = parser.parse_args()

    assert args.method is not None
    print("\nArguments:")
    print("=" * 40)
    for arg in vars(args):
        print(f"{arg: <20}: {getattr(args, arg)}")
    print("=" * 40)

    if args.parallel:
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"Start Multi-GPU Training with GPU Num {num_gpus} !!!")
        # training_parallel(args=args, num_gpus=num_gpus)
    else:
        training(args=args)
        print(f"Start Single-GPU Training !!!")
