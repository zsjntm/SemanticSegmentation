from losses.loss_functions import SSCrossEntropy
from datasets.cityscapes import Cityscapes
import numpy as np

def build_loss(*args):
    """
    :param args[0]: ignore_idx
    :param args[1]: reduction: 0->'mean' 1->'sum'
    :param args[2]: ohem 0->不用 1->使用
    :param args[3]: thres_p: 概率低于该值的被认为难样本, 仅在开启ohem时有效
    :param args[4]: min_kept, 仅在开启ohem时有效
    :param args[5]: class_weights -1->不用, 1->用cityscapes的
    :param args[6]: -1->'cpu', i>=0->'cuda:i'
    """

    ignore_idx = int(args[0])

    reduction = 'mean' if args[1] == 0 else 'sum'

    if args[2] == 0:
        ohem = None
    else:
        thres = -np.log(args[3])
        ohem = (thres, args[4])

    if args[5] == -1:
        class_weights = None
    elif args[5] == 1:
        class_weights = Cityscapes.class_weight

    device = 'cuda:{}'.format(args[6]) if args[6] >= 0 else 'cpu'

    lf = SSCrossEntropy(ignore_idx, reduction, ohem, class_weights, device)
    return lf

