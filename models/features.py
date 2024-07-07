try:
    from __init__ import torchvision, nn
    from modules import BasicBlock
except:
    from .__init__ import torchvision, nn
    from .modules import BasicBlock


def get_r18_feature(pretrained=True, stage_final_activation=True):
    """
    返回resnet18的特征提取部分
    :param pretrained: 是否使用IN-1K预训练的权重
    :param stage_final_activation: 每个stage的输出是否使用最终的relu激活
    """
    if pretrained:
        r18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        r18 = torchvision.models.resnet18()

    class resnet18_feature(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(r18.conv1, r18.bn1, r18.relu, r18.maxpool)
            self.layer1 = nn.Sequential(r18.layer1[0], BasicBlock(64, False))
            self.layer2 = nn.Sequential(r18.layer2[0], BasicBlock(128, False))
            self.layer3 = nn.Sequential(r18.layer3[0], BasicBlock(256, False))
            self.layer4 = nn.Sequential(r18.layer4[0], BasicBlock(512, False))
            self.relu = nn.ReLU()

            if pretrained:
                self.layer1.load_state_dict(r18.layer1.state_dict())
                self.layer2.load_state_dict(r18.layer2.state_dict())
                self.layer3.load_state_dict(r18.layer3.state_dict())
                self.layer4.load_state_dict(r18.layer4.state_dict())

            self.stage_final_activation = stage_final_activation

        def forward(self, x):
            if self.stage_final_activation:
                return self.__forward_v1(x)
            else:
                return self.__forward_v2(x)

        def __forward_v1(self, x):
            x1 = self.relu(self.layer1(self.stem(x)))
            x2 = self.relu(self.layer2(x1))
            x3 = self.relu(self.layer3(x2))
            x4 = self.relu(self.layer4(x3))
            return x1, x2, x3, x4

        def __forward_v2(self, x):
            x1 = self.layer1(self.stem(x))
            x2 = self.layer2(self.relu(x1))
            x3 = self.layer3(self.relu(x2))
            x4 = self.layer4(self.relu(x3))
            return x1, x2, x3, x4

    return resnet18_feature()


if __name__ == '__main__':
    import torch

    feature = get_r18_feature(True, False)
    print(feature)
    x = torch.rand(3, 3, 224, 224)
    x1, x2, x3, x4 = feature(x)
    print(x1.shape, x2.shape, x3.shape, x4.shape)
    print((x1 < 0).sum(), (x2 < 0).sum(), (x3 < 0).sum(), (x4 < 0).sum())
