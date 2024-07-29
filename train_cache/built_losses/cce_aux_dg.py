from train_cache.built_losses import cce
from losses.loss_functions import weighted_bce
from losses.loss_functions import SSCrossEntropy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Target2BDTarget:
    """用来生成细节引导的标签，生成的标签为二值，dtype为float32"""

    def __init__(self, device):
        self.device = device

        # 三个laplacian算子，不需要梯度
        self.laplacianx1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)  # 填充为1，便于生成原尺寸
        self.laplacianx1.weight = nn.Parameter(
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32,
                         device=self.device).contiguous().view(1, 1, 3, 3),
            requires_grad=False)

        self.laplacianx2 = nn.Conv2d(1, 1, 3, 2, bias=False)
        self.laplacianx2.weight = nn.Parameter(
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32,
                         device=self.device).contiguous().view(1, 1, 3, 3),
            requires_grad=False)

        self.laplacianx4 = nn.Conv2d(1, 1, 3, 4, bias=False)
        self.laplacianx4.weight = nn.Parameter(
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32,
                         device=self.device).contiguous().view(1, 1, 3, 3),
            requires_grad=False)

    def __laplacian_process(self, target, kernel):
        """
        获取标签中不同类别的边界
        :param target: original_gt: tensor (b, 1, h, w) int64
        :return: (b, 1, h', w') float32 只有0，1两个值
        """

        bd = kernel(target.float())  # tensor (b, 1, h', w')
        bd = torch.abs(bd) > 0  # 绝对值化后，二值化
        return bd.float()

    def __call__(self, target):
        """
        将target进行不同尺度的边缘提取，并将不同尺度的边界信息直接相加
        :param target: original_gt: (b, 1, h, w) int64
        :return: bd_gt: (b, 1, h, w) float32 只有0，1两个值
        """
        _, _, h, w = target.shape
        bd1x = self.__laplacian_process(target, self.laplacianx1)
        bd2x = self.__laplacian_process(target, self.laplacianx2)
        bd4x = self.__laplacian_process(target, self.laplacianx4)
        bd = bd1x + F.interpolate(bd2x, (h, w), mode='bilinear') + F.interpolate(bd4x, (h, w), mode='bilinear')
        return (bd > 0).type(torch.float32)  # 二值化，再进行类型转换


class Loss:
    def __init__(self, lf, lf_aux, lf_detail, aux_weight, detail_weight, device):
        self.lf = lf
        self.lf_aux = lf_aux
        self.lf_detail = lf_detail
        self.aux_weight = aux_weight
        self.detail_weight = detail_weight
        self.ss2bd = Target2BDTarget(device)

    def __call__(self, outputs, targets):
        """
        :param outputs: (final_seg, aux_seg, detail_seg)
        """

        loss_final = self.lf(outputs[0], targets)
        loss_aux = self.lf_aux(outputs[1], targets)
        bd_targets = self.ss2bd(targets)
        loss_detail = self.lf_detail(outputs[2], bd_targets)

        return loss_final + loss_aux * self.aux_weight + loss_detail * self.detail_weight


def build_loss(*args):
    """
    :param args[0~6]: cce的build_loss所需的参数
    :param args[7]: aux_loss的ignore_idx
    :param args[8]: aux_loss的reduction: 0->'mean' 1->'sum'
    :param args[9]: aux_loss的aux_weight
    :param args[10]: detail_loss的detail_weight
    :param args[11]: detail_loss的device: -1->'cpu' i>=0->'cuda:i'
    """

    lf_final = cce.build_loss(*args[:7])

    ignore_idx = int(args[7])
    reduction = 'mean' if args[8] == 0 else 'sum'
    aux_weight = args[9]
    lf_aux = SSCrossEntropy(ignore_idx, reduction)

    lf_detail = weighted_bce
    detail_weight = args[10]
    device = 'cuda:{}'.format(int(args[11])) if args[11] >= 0 else 'cpu'
    return Loss(lf_final, lf_aux, lf_detail, aux_weight, detail_weight, device)


if __name__ == '__main__':
    from tools.model_tools import load_model

    model = load_model(r'../built_models/cityscapes_models/sota/eudr1_64x_aux_dg')

    from train_cache.built_datasets import cityscapes_train_recipe1

    dataset = cityscapes_train_recipe1.build()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, 3)
    for imgs, targets in dataloader:
        break
    imgs, targets = imgs.to('cuda'), targets.to('cuda')

    model.train().to('cuda')
    outputs = model(imgs)
    loss = build_loss(255, 0, 1, 0.9, 131072, -1, 0, 255, 0, 1, 1, 0)
    print(loss(outputs, targets))
