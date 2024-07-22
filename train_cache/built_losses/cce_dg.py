from train_cache.built_losses import cce
from losses.loss_functions import weighted_bce


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
    def __init__(self, lf_final, lf_detail, detail_weight, device):
        self.lf_final = lf_final
        self.lf_detail = lf_detail
        self.detail_weight = detail_weight
        self.t2bdt = Target2BDTarget(device)
        self.device = device

    def __call__(self, outputs, targets):
        """
        :param outputs: (final_outputs, detail_outputs)
        :param targets: targets
        :return: loss值
        """

        loss_final = self.lf_final(outputs[0], targets)

        bd_targets = self.t2bdt(targets)  # (b, 1, h, w) float32 只有0，1两个值
        loss_detail = self.lf_detail(outputs[1], bd_targets)

        return loss_final + self.detail_weight * loss_detail

def build_loss(*args):
    """
    :param args[0~6]: cce的build_loss所需的参数
    :param args[7]: detail_loss的detail_weight
    :param args[8]: detail_loss的device: -1->'cpu' i>=0->'cuda:i'
    """

    lf_final = cce.build_loss(*args[:7])

    lf_detail = weighted_bce
    detail_weight = args[7]
    device = 'cuda:{}'.format(args[8]) if args[8] >= 0 else 'cpu'

    lf = Loss(lf_final, lf_detail, detail_weight, device)
    return lf


