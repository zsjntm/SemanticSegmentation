from datasets.cityscapes import Cityscapes
from datasets.train_transforms import VanillaTransform
from env import CITYSCAPES_DIR


def build():
    return Cityscapes(CITYSCAPES_DIR, 'train', VanillaTransform(1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))


if __name__ == '__main__':
    dataset = build()
    print(len(dataset))
    img, target = dataset[0]
    print(img.shape, img.dtype)
    print(target.shape, target.dtype)
