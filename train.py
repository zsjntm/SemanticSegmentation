import argparse
import time

from tools.model_tools import load_model
from torch import optim
from train.trainer import SemanticSegmentationTrainer
from train.lr_schedulers import DADLRScheduler, NOLRScheduler


def get_args():
    parser = argparse.ArgumentParser(description='=======train args=======')
    parser.add_argument('--model_dir', type=str, dest='model_dir')
    parser.add_argument('--dataset', type=str, dest='dataset')
    parser.add_argument('--lf', type=str, default='cce', help='default=cce', dest='lf')  # 默认用cce
    parser.add_argument('--lf_args', nargs='+', type=float, default=[255, 0], help='default=[255, 0]',
                        dest='lf_args')  # 默认用255, 'mean'
    parser.add_argument('--evaluator', type=str, dest='evaluator')
    parser.add_argument('--eval_bsize_nw', nargs='+', type=int, dest='eval_bsize_nw')
    parser.add_argument('--checkpoint', type=str, dest='checkpoint')

    # optimizer, lr_scheduler
    parser.add_argument('--optimizer', type=str, default='SGD', help='default=SGD', dest='optimizer')  # 只支持SGD
    parser.add_argument('--lr', type=float, dest='lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='default=0.9', dest='momentum')
    parser.add_argument('--wd', type=float, default=0, help='default=0', dest='wd')
    parser.add_argument('--lr_scheduler', type=str, default='none', help='default=none',
                        dest='lr_scheduler')  # 只支持DAD，none
    parser.add_argument('--ls_args', nargs='+', type=float, dest='ls_args')

    # 其他训练配置
    parser.add_argument('--bsize', type=int, dest='bsize')
    parser.add_argument('--num_workers', type=int, dest='num_workers')
    parser.add_argument('--no_shuffle', action='store_true', dest='no_shuffle')
    parser.add_argument('--no_use_autocast', action='store_true', dest='no_use_autocast')
    parser.add_argument('--val_interval', type=int, default=5, help='default=5', dest='val_interval')

    # start参数
    parser.add_argument('--epoch', type=int, dest='epoch')
    parser.add_argument('--device', type=str, default='cuda', help='default is cuda', dest='device')
    parser.add_argument('--verbose', type=int, default=1, help='default=1', dest='verbose')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 从外部接受参数
    args = get_args()
    all_t = time.time()

    # 1.接受从命令行传入的参数
    model_dir = args.model_dir
    dataset_name = args.dataset
    lf_name, lf_args = args.lf, args.lf_args
    evaluator_name, eval_bsize_nw = args.evaluator, args.eval_bsize_nw
    checkpoint = args.checkpoint

    optimizer_name, lr, momentum, wd = args.optimizer, args.lr, args.momentum, args.wd
    lr_scheduler_name, ls_args = args.lr_scheduler, args.ls_args
    bsize, num_workers, shuffle, use_autocast, val_interval = args.bsize, args.num_workers, not args.no_shuffle, not args.no_use_autocast, args.val_interval
    epoch, device, verbose = args.epoch, args.device, args.verbose

    # 2.构建model, optimizer, lr_scheduler
    model = load_model(model_dir, device=device)
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    if lr_scheduler_name == 'none':  # 不需要ls_args
        lr_scheduler = NOLRScheduler(optimizer)
    elif lr_scheduler_name == 'DAD':  # ls_args为[max_iter, power]
        lr_scheduler = DADLRScheduler(optimizer, *ls_args)

    # 3.构建dataset, lf, evaluator
    exec('from train_cache.built_datasets.{} import build as build_dataset'.format(dataset_name))
    exec('from train_cache.built_losses.{} import build_loss'.format(lf_name))
    exec('from train_cache.built_evaluators.{} import build_evaluator'.format(evaluator_name))
    dataset = build_dataset()
    lf = build_loss(*lf_args)
    evaluator = build_evaluator(*eval_bsize_nw)

    # 4.构建trainer
    trainer = SemanticSegmentationTrainer(model, dataset, lf, optimizer, lr_scheduler, evaluator,
                                          bsize, num_workers, shuffle, use_autocast, val_interval, checkpoint)

    # 启动
    trainer.start(epoch, device, verbose)
    print('train.py exec end, spend time:{:.2f}min'.format((time.time() - all_t) / 60))
