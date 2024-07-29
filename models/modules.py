try:
    from .__init__ import torch, nn, F, OrderedDict
except:
    from __init__ import torch, nn, F, OrderedDict


def get_norm_activation_conv(in_channels, out_channels, ksize, stride=1, padding=0, norm_type=nn.BatchNorm2d,
                             activation_type=nn.ReLU):
    lis = [
        ('norm', norm_type(num_features=in_channels)),
        ('activation', activation_type(inplace=True)),
        ('conv', nn.Conv2d(in_channels, out_channels, ksize, stride, padding)),
    ]
    return nn.Sequential(OrderedDict(lis))


def get_conv_norm(in_channels, out_channels, ksize, stride=1, padding=0, norm_type=nn.BatchNorm2d):
    lis = [('conv', nn.Conv2d(in_channels, out_channels, ksize, stride, padding, bias=False)),
           ('norm', norm_type(num_features=out_channels))]
    return nn.Sequential(OrderedDict(lis))


def get_conv_norm_activation(in_channels, out_channels, ksize, stride=1, padding=0, norm_type=nn.BatchNorm2d,
                             activation_type=nn.ReLU):
    result = get_conv_norm(in_channels, out_channels, ksize, stride, padding, norm_type)
    result.add_module('activation', activation_type(inplace=True))
    return result


class SegModel(nn.Module):
    """作为其他分割模型的父类，只有一个分割头"""

    def __init__(self, seg_channels, cls_n, seg_head='default'):
        """
        :param seg_channels: 最终用来分割的特征的通道数
        :param cls_n:
        :param seg_head: option: 'default': conv1x1; 或其他分割头
        """
        super().__init__()
        if seg_head == 'default':
            self.seg_head = nn.Conv2d(seg_channels, cls_n, 1)
        else:
            self.seg_head = seg_head

    def forward(self, seg_feature, x_size=None):
        """
        :param seg_feature: 输入分割头的特征
        :return: 若x_size为(h, w), 则会将输出的logits双线性上采样到输入尺寸
        """

        logits = self.seg_head(seg_feature)
        if x_size is not None:
            return F.interpolate(logits, size=x_size, mode='bilinear', antialias=True)
        else:
            return logits


class SFFM(nn.Module):
    """最原始的版本,两个cbr，无多余结构"""

    def __init__(self, spatial_in_channels, spatial_out_channels, context_channels, out_channels):
        super().__init__()

        self.conv_spatial = get_conv_norm_activation(spatial_in_channels, spatial_out_channels, 3, 1, 1)
        self.conv_fuse = get_conv_norm_activation(context_channels + spatial_out_channels, out_channels, 3, 1, 1)

    def forward(self, spatial, context):
        _, _, h, w = spatial.shape
        spatial = self.conv_spatial(spatial)
        return self.conv_fuse(torch.cat([spatial, F.interpolate(context, (h, w), mode='bilinear')], dim=-3))


class ECM(nn.Module):
    """最原始的版本,两个cbr，无多余结构"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_global = get_conv_norm_activation(in_channels, in_channels, 1)
        self.conv_fuse = get_conv_norm_activation(in_channels * 2, out_channels, 3, 1, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        context = self.conv_global(self.gap(x))
        return self.conv_fuse(torch.cat([x, F.interpolate(context, (h, w), mode='nearest')], dim=-3))


class PPM(nn.Module):
    """经典的PPM后面多了个cbr3"""

    def __init__(self, in_channels, out_channels, bins=[1, 2, 3, 6]):
        super().__init__()

        branches_n = len(bins)
        branches = []
        for bin in bins:
            lis = [
                ('pool', nn.AdaptiveAvgPool2d(bin)),
                ('cbr', get_conv_norm_activation(in_channels, in_channels // branches_n, 1))
            ]
            branches.append(nn.Sequential(OrderedDict(lis)))
        self.branches = nn.ModuleList(branches)
        self.conv_fuse = get_conv_norm_activation(in_channels + (in_channels // branches_n) * branches_n, out_channels,
                                                  3, 1, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        branches_feats = [x]
        for branch in self.branches:
            branches_feats.append(F.interpolate(branch(x), (h, w), mode='bilinear', align_corners=True, antialias=True))
        return self.conv_fuse(torch.cat(branches_feats, dim=-3))


class BasicBlock(nn.Module):
    """Res18的block，可以不要最终的relu激活，可以二倍下采样, 与torch的BasicBlock同步"""

    def __init__(self, channels, output_relu=False, downsample=False, in_channels=None):
        super().__init__()
        if in_channels is None:
            in_channels = channels

        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if downsample == True:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, 2, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.conv1.stride = 2
        else:
            self.downsample = None

        self.output_relu = output_relu

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual
        if self.output_relu:
            return self.relu(x)
        else:
            return x


class BasicBlockV2(nn.Module):
    """Res18V2的恒等形状输出block"""

    def __init__(self, channels):
        super().__init__()
        self.brc1 = get_norm_activation_conv(channels, channels, 3, 1, 1)
        self.brc2 = get_norm_activation_conv(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.brc2(self.brc1(x))


if __name__ == '__main__':
    '''get_conv_norm_activation'''
    # cbr = get_conv_norm_activation(3, 64, 3, stride=1, padding=1)
    # print(cbr)
    # x = torch.rand(1,3,224,224)
    # print(cbr(x).shape)

    '''ECM'''
    # ecm = ECM(512, 256)
    # x = torch.rand(3, 512, 7, 7)
    # print(ecm)
    # print(ecm(x).shape)

    '''PPM'''
    # ppm = PPM(512, 256)
    # x = torch.rand(3, 512, 7, 7)
    # print(ppm)
    # print(ppm(x).shape)

    "BasicBlock"
    # import torchvision
    #
    # r18 = torchvision.models.resnet18()
    # print(r18)
    #
    # BBD = BasicBlock(64, True)
    # print(BBD)
    # x = torch.rand(3, 64, 56, 56)
    # y = BBD(x)
    # print(y.shape)
    #
    # BBD = BasicBlock(128, True, True, 64)
    # print(BBD)
    # x = torch.rand(3, 64, 56, 56)
    # y = BBD(x)
    # print(y.shape)
