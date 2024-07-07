"""包含一些经过transforms中的各个零件组装成的复合型transform，主要用于训练"""

try:
    from transforms import *
except:
    from .transforms import *


class VanillaTransform:
    def __init__(self, size, mean, std):
        self.transform = Compose([
            ToImage(),
            RandomResizedCrop(size),
            ToDtype(torch.float32, torch.int64),
            Normalize(mean, std),
        ])

    def __call__(self, img, target):
        return self.transform(img, target)


class TrainTransformV1:
    def __init__(self, size, mean, std, scale=(0.08, 1), ratio=(3 / 4, 4 / 3), flip_p=0.5):
        self.transform = Compose([
            ToImage(),
            RandomResizedCrop(size, scale, ratio),
            RandomHorizontalFlip(flip_p),
            ToDtype(torch.float32, torch.int64),
            Normalize(mean, std)
        ])

    def __call__(self, img, target):
        return self.transform(img, target)

class TrainTransformV2:
    def __init__(self, size, mean, std, scale=(0.08, 1), ratio=(3 / 4, 4 / 3), rotation_range=10, flip_p=0.5):
        self.transform = Compose([
            ToImage(),
            RandomResizedCrop(size, scale, ratio),
            RandomRotation(rotation_range),
            RandomHorizontalFlip(flip_p),
            ToDtype(torch.float32, torch.int64),
            Normalize(mean, std)
        ])

    def __call__(self, img, target):
        return self.transform(img, target)