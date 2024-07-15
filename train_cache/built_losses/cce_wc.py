import torch
import torch.nn.functional as F
from datasets.cityscapes import Cityscapes


def build_loss(*args):
    """
    :param args[0]: border_idx
    :param args[1]: reduction: 0->'mean', 1->'sum'
    :param args[2]: 采用哪个数据集的class_weight, 1->'cityscapes'
    :param args[3]: 训练所在设备, -1->'cpu', 0->'cuda:0', 1->'cuda:1'
    """


    border_idx = int(args[0])

    assert args[1] == 0 or args[1] == 1
    reduction = 'mean' if args[1] == 0 else 'sum'

    if args[2] == 1:
        class_weight = Cityscapes.class_weight

    assert args[3] == -1 or args[3] >= 0
    if args[3] == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(int(args[3]))

    class Loss:
        def __init__(self, ignore_idx, reduction, class_weight, device='cuda'):
            """
            :param class_weight: list 里面包含每个类的权重
            """
            self.ignore_idx = ignore_idx
            self.reduction = reduction
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)

        def __call__(self, outputs, targets):
            """
            :param outputs: (b, c, h, w)
            :param targets: (b, 1, h, w)
            """
            b, c, h, w = outputs.shape
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, c)  # （b * h * w, c)
            targets = targets.contiguous().view(-1)  # (b * h * w, )

            return F.cross_entropy(outputs, targets, self.class_weight, ignore_index=self.ignore_idx,
                                   reduction=self.reduction)

    return Loss(border_idx, reduction, class_weight, device)
