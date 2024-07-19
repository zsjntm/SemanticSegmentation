import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    """
    cce + bd的bce
    """
    def __init__(self, ignore_idx, reduction, detail_weight, device):
        self.ignore_idx = ignore_idx
        self.reduction = reduction
        self.detail_weight = detail_weight
        self.device = device

        # 三个laplacian算子，不需要梯度
        self.laplacianx1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)  # 填充为1，便于生成原尺寸
        self.laplacianx1.weight = nn.Parameter(
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32, device=self.device).contiguous().view(1, 1, 3, 3),
            requires_grad=False)

        self.laplacianx2 = nn.Conv2d(1, 1, 3, 2, bias=False)
        self.laplacianx2.weight = nn.Parameter(
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32, device=self.device).contiguous().view(1, 1, 3, 3),
            requires_grad=False)

        self.laplacianx4 = nn.Conv2d(1, 1, 3, 4, bias=False)
        self.laplacianx4.weight = nn.Parameter(
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32, device=self.device).contiguous().view(1, 1, 3, 3),
            requires_grad=False)

    def __laplacian_process(self, target, kernel):
        """
        获取标签中不同类别的边界
        :param target: original_gt: tensor (b, 1, h, w) int64
        :return: (b, 1, h', w') float32
        """

        bd = kernel(target.float())  # tensor (b, 1, h', w')
        bd = torch.abs(bd) > 0  # 绝对值化后，二值化
        return bd.float()

    def get_bd_gt(self, target):
        """
        将不同尺度的边界信息直接相加
        :param target: original_gt: (b, 1, h, w) int64
        :return: bd_gt: (b, 1, h, w) float32 只有0，1两个值
        """
        _, _, h, w = target.shape
        bd1x = self.__laplacian_process(target, self.laplacianx1)
        bd2x = self.__laplacian_process(target, self.laplacianx2)
        bd4x = self.__laplacian_process(target, self.laplacianx4)
        bd = bd1x + F.interpolate(bd2x, (h, w), mode='bilinear') + F.interpolate(bd4x, (h, w), mode='bilinear')
        return (bd > 0).type(torch.float32)  # 二值化，再进行类型转换

    def __cce(self, outputs, targets):
        """
        :param outputs: (b, cls_n, h, w) float32 logits
        :param targets: (b, 1, h, w) int64 cls_idx
        """

        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.shape[-3])  # (b * h * w, cls_n)
        targets = targets.flatten().contiguous()  # (b * h * w, )
        return F.cross_entropy(outputs, targets, ignore_index=self.ignore_idx, reduction=self.reduction)

    def __weighted_bce(self, outputs, bd_gts):
        """
        pos的权重为neg的数量的百分比，neg同理
        :param outputs: (b, 1, h, w) float32 logits
        :param bd_gts: (b, 1, h, w) float32 只有0，1两个值
        """
        outputs, bd_gts = outputs.view(-1), bd_gts.view(-1)  # (bhw, ), (bhw, )

        pos_mask, neg_mask = (bd_gts == 1), (bd_gts == 0)
        pos_num, neg_num = pos_mask.sum(), neg_mask.sum()
        pos_weight, neg_weight = neg_num / (pos_num + neg_num), pos_num / (pos_num + neg_num)
        weight = torch.zeros_like(outputs, dtype=torch.float32, device=self.device)  # 注意，这里用了float32的数据类型，
        weight[pos_mask], weight[neg_mask] = pos_weight, neg_weight

        return F.binary_cross_entropy_with_logits(outputs, bd_gts, weight, reduction='mean')

    def __call__(self, outputs, targets):
        """
        :param outputs: final_seg (b, cls_n, h, w) logits, detail_seg (b, 1, h, w) logits
        :param targets: fianl_target (b, 1, h, w) cls_idx
        """
        bd_gts = self.get_bd_gt(targets)  # (b, 1, h, w) float32 只有0，1两个值
        bd_gts = bd_gts.to(outputs[1].device).type(outputs[1].dtype)
        cce_loss = self.__cce(outputs[0], targets)
        bce_loss = self.__weighted_bce(outputs[1], bd_gts)
        return cce_loss + self.detail_weight * bce_loss

def build_loss(*args):
    """
    :param args[0]: ignore_idx
    :param args[1]: reduction: 0->'mean', 1->'sum'
    :param args[2]: detail_weight
    :param args[3]: device: -1->'cpu', 0->'cuda:0', 1->'cuda:1'
    """

    ignore_idx = int(args[0])
    reduction = 'mean' if args[1] == 0 else 'sum'
    detail_weight = args[2]
    device = 'cuda:{}'.format(int(args[3])) if args[3] >= 0 else 'cpu'

    loss = Loss(ignore_idx, reduction, detail_weight, device)
    return loss



if __name__ == '__main__':
    from train_cache.built_datasets import cityscapes_val
    dataset = cityscapes_val.build()
    _, target = dataset[0]
    target = target.unsqueeze(0).to('cuda')
    outputs = (torch.rand(1, 19, 1024, 2048).to('cuda'), torch.rand(1, 1, 1024, 2048).to('cuda'))

    loss = build_loss(*(255, 0, 20, 0))
    print(loss(outputs, target))

