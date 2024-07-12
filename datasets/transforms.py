"""包含一些transform的零件，用Compose来组成复杂的transform"""

try:
    from .__init__ import v2, torch, np
except:
    from __init__ import v2, torch, np


class ToImage:
    def __init__(self):
        self.to_image = v2.ToImage()

    def __call__(self, img, target):
        """
        :param img: PIL
        :param target: PIL
        :return: tv_tensor: (3, h, w) uint8, tv_tensor: (1, h, w) uint8
        """
        img = self.to_image(img)
        target = self.to_image(target)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        """
        :param p: 执行水平翻转的概率
        """
        self.p = p

    def __call__(self, img, target):
        if np.random.choice(2, p=[1 - self.p, self.p]) == 1:
            img = v2.functional.horizontal_flip(img)
            target = v2.functional.horizontal_flip(target)
        return img, target


class RandomRotation:
    def __init__(self, rotation_range=10, fill_value=255):
        """
        :param fill_value: 标签的填充值
        """
        self.fill_value = fill_value
        if type(rotation_range) is int:
            rotation_range = (-rotation_range, rotation_range)
        self.rotation_range = rotation_range

    def __call__(self, img, target):
        rotation = np.random.uniform(*self.rotation_range)
        img = v2.functional.rotate(img, rotation, v2.InterpolationMode.BILINEAR)
        target = v2.functional.rotate(target, rotation, v2.InterpolationMode.NEAREST, fill=self.fill_value)
        return img, target


class Resize:
    def __init__(self, size):
        """
        :param size: 为int将短边对齐，为tuple将resize到目标尺寸
        """
        self.resize_img = v2.Resize(size, antialias=True)
        self.resize_target = v2.Resize(size, v2.InterpolationMode.NEAREST)

    def __call__(self, img, target):
        img = self.resize_img(img)
        target = self.resize_target(target)
        return img, target

class RandomResize:
    def __init__(self, scale_range=(0.5, 2.1)):
        """
        :param scale_range: [low_scale_factor, high_scale_factor)
        """
        self.scale_range = scale_range

    def __call__(self, img, target):
        _, h, w = img.shape
        factor = np.random.uniform(*self.scale_range)
        h, w = int(h * factor), int(w * factor)
        img = v2.functional.resize(img, (h, w), antialias=True)
        target = v2.functional.resize(target, (h, w), interpolation=v2.InterpolationMode.NEAREST)
        return img, target

class Pad2Dsize:
    def __init__(self, dsize, fill_value=255):
        """
        将img，target不满dsize的边填充到dsize的长度
        :param dsize: (h, w)
        :param fill_value: 标签的填充值
        """
        self.dsize = dsize
        self.fill_value = fill_value

    def __call__(self, img, target):
        _, h, w = img.shape
        pad_h, pad_w = max(self.dsize[0] - h, 0), max(self.dsize[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = v2.functional.pad(img, (0, 0, pad_w, pad_h), fill=0)
            target = v2.functional.pad(target, (0, 0, pad_w, pad_h), fill=self.fill_value)
        return img, target

class RandomCrop:
    def __init__(self, crop_size):
        """
        :param crop_size: (h, w)
        """
        self.crop_size = crop_size

    def __call__(self, img, target):
        top, left, h, w = v2.RandomCrop.get_params(img, self.crop_size)
        img = v2.functional.crop(img, top, left, h, w)
        target = v2.functional.crop(target, top, left, h, w)
        return img, target



class RandomResizedCrop:
    def __init__(self, size=1024, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
        self.random_resized_crop = v2.RandomResizedCrop(size, scale=scale, ratio=ratio, antialias=True)

    def __call__(self, img, target):
        rec = self.random_resized_crop.get_params(img,
                                                  scale=self.random_resized_crop.scale,
                                                  ratio=self.random_resized_crop.ratio)
        img = v2.functional.resized_crop(img, *rec, size=self.random_resized_crop.size, antialias=True)
        target = v2.functional.resized_crop(target, *rec, size=self.random_resized_crop.size,
                                            interpolation=v2.InterpolationMode.NEAREST, antialias=True)
        return img, target


class ToDtype:
    def __init__(self, img_dtype, target_dtype):
        self.to_dtype_img = v2.ToDtype(img_dtype, scale=True)
        self.to_dtype_target = v2.ToDtype(target_dtype, scale=False)

    def __call__(self, img, target):
        """
        :param img: 到目标dtype并scale
        :param target: 到目标dtype
        :return:
        """
        img = self.to_dtype_img(img)
        target = self.to_dtype_target(target)
        return img, target


class Normalize:
    """只对img标准化"""

    def __init__(self, mean, std):
        self.normalize = v2.Normalize(mean, std, True)

    def __call__(self, img, target):
        img = self.normalize(img)
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for transform in self.transforms:
            img, target = transform(img, target)
        return img, target


if __name__ == '__main__':
    import torch
    from cityscapes import Cityscapes

    dataset = Cityscapes(r'../data/cityscapes', 'train')
    img, target = dataset[0]

    # to_image = ToImage()
    # img, target = to_image(img, target)
    #
    # random_resized_crop = RandomResizedCrop()
    # img, target = random_resized_crop(img, target)
    #
    # to_dtype = ToDtype(torch.float32, torch.int64)
    # img, target = to_dtype(img, target)
    #
    # transform = VanillaTransform(1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # img, target = transform(img, target)
    #
    # print(img.shape, img.dtype, type(img), (img.min(), img.max()))
    # print(target.shape, target.dtype, type(target), (target.min(), target.max()))
    #
    # from tools.img_process import Tensor2img
    # import matplotlib.pyplot as plt
    #
    # tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # img = tensor2img(img)
    # print(img.shape, img.dtype, type(img), (img.min(), img.max()))
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(target.numpy()[0])
    # plt.show()
