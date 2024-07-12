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


class TrainTransformV3:
    def __init__(self, size, mean, std, scale_range, fill_value, rotation_range=10, flip_p=0.5):
        """
        :param size: (h, w)
        :param scale_range: [low_scale_factor, high_scale_factor)
        :param fill_value: 标签的填充值
        """
        self.transform = Compose([
            ToImage(),
            RandomResize(scale_range),
            Pad2Dsize(size, fill_value),
            RandomCrop(size),
            RandomRotation(rotation_range, fill_value),
            RandomHorizontalFlip(flip_p),
            ToDtype(torch.float32, torch.int64),
            Normalize(mean, std),
        ])

    def __call__(self, img, target):
        return self.transform(img, target)
