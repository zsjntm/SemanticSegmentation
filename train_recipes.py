"""这些recipes会确定训练函数中固定的一部分参数，剩下的参数需要自己填"""
import os

from env import TRAIN_PY_PATH

VOC2012_DEFAULT = {
    'dataset': 'voc2012_train',
    'evaluator': 'voc2012_val_cce',
    'eval_bsize_nw': [256, 1],
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'wd': 0,
    'lr_scheduler': 'none',
    'ls_args': 0,
    'val_interval': 3,
}  # R18系列的分割网络，在voc2012上训练用这个，bsize大到跑满GPU，nw大到内存不爆，epoch随意

CITYSCAPES_DEFAULT = {
    'dataset': 'cityscapes_train_recipe1',
    'evaluator': 'cityscapes_val_cce',
    'eval_bsize_nw': [20, 1],
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'wd': 0,
    'lr_scheduler': 'none',
    'ls_args': 0,
    'val_interval': 5,
}  # R18系列的分割网络，在cityscapes上训练用这个，bsize大到跑满GPU，nw大到内存不爆，epoch随意

# 一个recipe分为两个阶段，两个一起用
CITYSCAPES_DAD_STAGE1 = {
    'dataset': 'cityscapes_train_recipe3',
    'evaluator': 'cityscapes_val_cce',
    'eval_bsize_nw': [20, 1],
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'wd': 5e-4,
    'lr_scheduler': 'DAD',
    'ls_args': [120032, 0.9],
    'val_interval': 5,
}  # R18系列在cityscapes上要实现sota性能的recipe。stage1: bsize=12, epoch=384, nw大到内存不爆
CITYSCAPES_DAD_STAGE2 = {
    'dataset': 'cityscapes_train_recipe3',
    'evaluator': 'cityscapes_val_cce',
    'eval_bsize_nw': [20, 1],
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'wd': 5e-4,
    'lr_scheduler': 'DAD',
    'ls_args': [120032, 0.9],
    'val_interval': 1,
}  # stage2: bsize=12, epoch=100, nw大到内存不爆

# 一个recipe分为两个阶段，两个一起用
CITYSCAPES_DADV2_STAGE1 = {
    'dataset': 'cityscapes_train_dadv2',
    'evaluator': 'cityscapes_val_cce',
    'eval_bsize_nw': [20, 1],
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'wd': 5e-4,
    'lr_scheduler': 'DAD',
    'ls_args': [120032, 0.9],
    'val_interval': 5,
}  # R18系列在cityscapes上要实现sota性能的recipe。stage1: bsize=12, epoch=384, nw大到内存不爆
CITYSCAPES_DADV2_STAGE2 = {
    'dataset': 'cityscapes_train_dadv2',
    'evaluator': 'cityscapes_val_cce',
    'eval_bsize_nw': [20, 1],
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'wd': 5e-4,
    'lr_scheduler': 'DAD',
    'ls_args': [120032, 0.9],
    'val_interval': 1,
}  # stage2: bsize=12, epoch=100, nw大到内存不爆


def train(model_dir, checkpoint, bsize, nw, epoch, recipe, lf='cce', lf_args=[255, 0],
          device='cuda', verbose=1):
    def lis2str(lis):
        """
        将[1, 2, 0.3, ...] -> '1 2 0.3 ...'
        """
        return '{}'.format(lis).strip('[]').replace(',', '')

    def recipe2str(recipe):
        command = ''
        for k, v in recipe.items():
            if isinstance(v, (int, float, str)):
                command += '--{} {} '.format(k, v)
            if isinstance(v, list):  # 若v是list，那么每一项应该是int或float
                command += '--{} {} '.format(k, lis2str(v))
        return command

    command = (
        'python {} '
        '--model_dir {} '
        '--lf {} '
        '--lf_args {} '
        '--checkpoint {} '
        '--bsize {} '
        '--num_workers {} '
        '--epoch {} '
        '--device {} '
        '--verbose {} '.format(
            TRAIN_PY_PATH,
            model_dir,
            lf,
            lis2str(lf_args),
            checkpoint,
            bsize,
            nw,
            epoch,
            device,
            verbose,
        )
    )
    command2 = recipe2str(recipe)
    os.system(command + ' ' + command2)


if __name__ == '__main__':
    # train('model', 'checkpoint', 1, 0, 10, VOC2012_DEFAULT)
    train(r'train_cache/built_models/voc2012_models/r18',
          r'results/tmp',
          256, 2, 1,
          VOC2012_DEFAULT,
          )
