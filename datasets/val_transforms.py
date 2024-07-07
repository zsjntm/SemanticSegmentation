"""包含一些经过transforms中的各个零件组装成的复合型transform，主要用于评估"""

try:
    from transforms import *
except:
    from .transforms import *


class BaseTransform:
    def __init__(self, mean, std):
        self.transform = Compose([
            ToImage(),
            ToDtype(torch.float32, torch.int64),
            Normalize(mean, std),
        ])

    def __call__(self, img, target):
        return self.transform(img, target)


class ResizedBaseTransform:
    """在BaseTransform上增加了一步resize操作"""

    def __init__(self, size, mean, std):
        self.transform = Compose([
            ToImage(),
            Resize(size),
            ToDtype(torch.float32, torch.int64),
            Normalize(mean, std),
        ])

    def __call__(self, img, target):
        return self.transform(img, target)
