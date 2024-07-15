import torch
from datasets.cityscapes import Cityscapes
import torch.nn.functional as F
import numpy as np


def build_loss(*args):
    """
    先ignore筛选样本，再用ohem筛选样本，最后用class_weights将样本的损失加权和
    :param args[0]: border_idx
    :param args[1]: reduction: 0->'mean', 1->'sum'
    :param args[2]: 采用哪个数据集的class_weight，-1->不用, 0->'voc2012', 1->'cityscapes'  现只支持-1, 1
    :param args[3]: thres_p
    :param args[4]: min_kept
    :param args[5]: 训练所在设备, -1->'cpu', 0->'cuda:0', 1->'cuda:1'
    """

    # 参数翻译
    border_idx = int(args[0])
    assert args[1] == 0 or args[1] == 1
    reduction = 'mean' if args[1] == 0 else 'sum'
    assert args[2] == -1 or args[2] == 1
    if args[2] == -1:
        class_weights = None
    elif args[2] == 1:
        class_weights = Cityscapes.class_weight
    thres_p = args[3]
    min_kept = int(args[4])
    assert args[5] == -1 or args[5] >= 0
    if args[5] == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(int(args[5]))

    class Loss:
        def __init__(self, ignore_idx, reduction, class_weights, thres_p, min_kept, device):
            self.thres_loss = -np.log(thres_p)  # 由p的threshold得到loss的threshold，大于该损失被认为是难样本
            self.min_kept = min_kept  # 至少这么多的像素会参与反向传播
            self.ignore_idx = ignore_idx
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32,
                                              device=device) if class_weights is not None else None
            self.reduction = reduction

        def __call__(self, outputs, targets):
            """
            :param outputs: (b, c, h, w)
            :param targets: (b, 1, h, w)
            """

            b, c, h, w = outputs.shape
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, c)  # （b * h * w, c)
            targets = targets.contiguous().view(-1)  # (b * h * w, )

            # 1. 计算原始损失，并去掉ignore的像素
            losses = F.cross_entropy(outputs, targets, ignore_index=self.ignore_idx, reduction='none')  # (bhw, )
            ignore_mask = (targets != self.ignore_idx)
            losses = losses[ignore_mask]  # (B, )
            targets = targets[ignore_mask]  # (B, )

            # 2.预备好min_kept个像素
            sort_idx = torch.sort(losses, descending=True)[1][:self.min_kept]  # (min_kept, )
            min_kept_losses = losses[sort_idx]  # (min_kept, )
            min_kept_targets = targets[sort_idx]  # (min_kept, )

            # 3.按照阈值选择难样本
            mask = losses > self.thres_loss
            losses = losses[mask]  # (B', )
            targets = targets[mask]  # (B', )
            if len(losses) < self.min_kept:
                losses = min_kept_losses
                targets = min_kept_targets

            # 4.按照class_weight将损失加权和
            assert self.reduction == 'mean' or self.reduction == 'sum'
            if self.class_weights is not None:
                weights = self.class_weights[targets]  # (B', )
                if self.reduction == 'mean':
                    loss = (losses * weights).sum() / weights.sum()
                elif self.reduction == 'sum':
                    loss = (losses * weights).sum()
                return loss
            else:
                if self.reduction == 'mean':
                    loss = losses.mean()
                elif self.reduction == 'sum':
                    loss = losses.sum()
                return loss

    return Loss(border_idx, reduction, class_weights, thres_p, min_kept, device)


if __name__ == '__main__':
    pass
    # outputs = torch.rand(1, 21, 2, 3).to('cuda')
    # targets = torch.tensor([[1, 0, 255],
    #                         [1, 1, 1]]).contiguous().view(1, 1, 2, 3).to('cuda')
    #
    # lf = build_loss(255, 0, -1, 0.8, 100, 0)
    # lf(outputs, targets)
    # print(Cityscapes.border_index)
