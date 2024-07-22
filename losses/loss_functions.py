import torch
import torch.nn.functional as F


class OHEM:
    def __init__(self, thres, min_kept):
        """
        根据thres筛选(但保证至少有min_kept个样本),
        :param thres: 损失值的阈值，大于该值被认为难样本
        :param min_kept: 至少训练的样本的数量
        """
        self.thres = thres
        self.min_kept = min_kept

    def __call__(self, losses):
        """

        :param losses: (M, )
        :return: indices: (M', ) 难样本的索引
        """

        difficult_samples_indices = torch.where(losses > self.thres)[0]
        if len(difficult_samples_indices) < self.min_kept:  # 不足min_kept个，则返回前min_kept最大loss的索引
            return torch.sort(losses, descending=True)[1][:self.min_kept]  # (min_kept, )

        return difficult_samples_indices


class SSCrossEntropy:
    def __init__(self, ignore_idx, reduction, ohem=None, class_weights=None, device='cuda'):
        """
        用于语义分割的cce，可以附加上ohem、class_weights
        :param ignore_idx:
        :param reduction: 'mean', 'sum'
        :param ohem: None 或 (thres: 损失值的阈值 大于该值被认为难样本, min_kept: 至少训练的样本的数量)
        :param class_weights: list, 对应每个类别的权重
        :param device:
        """
        self.ignore_idx = ignore_idx
        self.reduction = reduction

        self.ohem = OHEM(ohem[0], ohem[1]) if ohem is not None else None
        self.class_weights = torch.tensor(class_weights, device=device) if class_weights is not None else None
        self.device = device

    def __call__(self, outputs, targets):
        """

        :param outputs: (b, cls_n, h, w) logits
        :param targets: (b, 1, h, w) indices
        :return: loss值
        """
        b, cls_n, h, w = outputs.shape
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, cls_n)  # (M, cls_n)
        targets = targets.view(-1)  # (M, )

        losses = F.cross_entropy(outputs, targets, ignore_index=self.ignore_idx, reduction='none')  # (M, )
        mask = targets != self.ignore_idx
        losses, targets = losses[mask], targets[mask]  # (M', ) 筛掉ignore的样本

        if self.ohem is not None:  # (M'', ) 用ohem筛选样本
            indices = self.ohem(losses)
            losses, targets = losses[indices], targets[indices]

        weights_sum = len(losses)
        if self.class_weights is not None:  # 根据class_weights加权
            weights = self.class_weights[targets]  # (M'', )
            losses = losses * weights  # (M'', ) 对损失加权
            weights_sum = weights.sum()

        if self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'mean':
            return losses.sum() / weights_sum

def weighted_bce(outputs, targets):
    """
    pos的权重为neg的数量的百分比，neg同理
    :param outputs: (b, 1, h, w) float32 logits
    :param targets: (b, 1, h, w) float32 只有0，1两个值
    :return: loss值
    """
    outputs, targets = outputs.view(-1), targets.view(-1)  # (bhw, ), (bhw, )

    pos_mask, neg_mask = (targets == 1), (targets == 0)
    pos_num, neg_num = pos_mask.sum(), neg_mask.sum()
    pos_weight, neg_weight = neg_num / (pos_num + neg_num), pos_num / (pos_num + neg_num)
    weight = torch.zeros_like(outputs, dtype=torch.float32, device=outputs.device)  # 注意，这里用了float32的数据类型，
    weight[pos_mask], weight[neg_mask] = pos_weight, neg_weight

    return F.binary_cross_entropy_with_logits(outputs, targets, weight, reduction='mean')

if __name__ == '__main__':
    """ohem"""
    # losses = torch.rand(5)
    # print(losses)
    # ohem = OHEM(0.5, 3)
    # print(ohem(losses))

    """SSCrossEntropy"""
    # outputs = torch.rand(3, 21, 224, 224)
    # targets = torch.randint(0, 20, (3, 1, 224, 224))
    # targets[:, 0, 20:100, 20:100] = 255
    # cce = SSCrossEntropy(255, 'mean', device='cpu')

    # 1
    # l1 = F.cross_entropy(outputs.permute(0, 2, 3, 1).contiguous().view(-1, 21), targets.view(-1), ignore_index=255,
    #                      reduction='mean')
    # l2 = cce(outputs, targets)
    # print(l1, l2)

    # 2
    # ohem = OHEM(0.5, 224*224*3)
    # cce = SSCrossEntropy(255, 'mean', ohem)
    # outputs = outputs - 0.6
    # print(outputs.min(), outputs.max())
    # l1 = F.cross_entropy(outputs.permute(0, 2, 3, 1).contiguous().view(-1, 21), targets.view(-1), ignore_index=255,
    #                      reduction='mean')
    # l2 = cce(outputs, targets)
    # print(l1, l2)

    # # 3
    # class_weights = torch.rand(21)
    # l1 = F.cross_entropy(outputs.permute(0, 2, 3, 1).contiguous().view(-1, 21), targets.view(-1), ignore_index=255,
    #                      reduction='mean', weight=class_weights)
    # cce = SSCrossEntropy(255, 'mean', None, class_weights.detach().clone().numpy(), 'cpu')
    # l2 = cce(outputs, targets)
    # print(l1, l2)

    # # 4
    # class_weights = torch.rand(21)
    # ohem = OHEM(0.5, 224*224*3)
    # outputs = outputs - 0.6
    #
    # l1 = F.cross_entropy(outputs.permute(0, 2, 3, 1).contiguous().view(-1, 21), targets.view(-1), ignore_index=255,
    #                      reduction='sum', weight=class_weights)
    # cce = SSCrossEntropy(255, 'sum', ohem, class_weights.detach().clone().numpy(), 'cpu')
    # l2 = cce(outputs, targets)
    # print(l1, l2)
