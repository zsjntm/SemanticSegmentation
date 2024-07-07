import torch.nn.functional as F


class Cross_Entorpy:
    def __init__(self, ignore_index=-100, reduction='mean'):
        """
        :param ignore_index: 这里的默认值为F.cross_entropy的ignore_index的默认值
        :param reduction: 这里的默认值为F.cross_entropy的reduction的默认值
        """
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, outputs, targets):
        """
        :param outputs: (b, cls_n, h, w) logits
        :param targets: (b, 1, h, w) indices
        :return:
        """
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.shape[-3])  # (b * h * w, cls_n)
        targets = targets.flatten().contiguous()  # (b * h * w, )
        loss = F.cross_entropy(outputs, targets, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


def build_loss(*args):
    """
    args[0]: ignore_index
    args[1]: reduction 0->'mean', 1->'sum'
    """
    ignore_index = int(args[0])  # float->int

    assert args[1] == 0 or args[1] == 1
    reduction = 'mean' if args[1] == 0 else 'sum'  # 0 -> 'mean', 1 -> 'sum'

    cce = Cross_Entorpy(ignore_index, reduction)
    return cce


if __name__ == '__main__':
    loss = build_loss(*[255., 1.])
    print(loss.ignore_index)
    print(loss.reduction)
