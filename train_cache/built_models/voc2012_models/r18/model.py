from models.features import get_r18_feature
from models.modules import SegModel


def build():
    class Net(SegModel):
        def __init__(self):
            super().__init__(512, 21)

            self.feature = get_r18_feature(True, True)

        def forward(self, x):
            b, c, h, w = x.shape
            x1, x2, x3, x4 = self.feature(x)
            return super().forward(x4, (h, w))
    return Net()


if __name__ == '__main__':
    import torch
    model = build()
    x = torch.rand(3, 3, 32, 32)
    print(model)
    print(model(x).shape)