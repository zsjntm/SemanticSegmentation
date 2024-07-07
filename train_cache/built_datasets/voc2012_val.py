from datasets.voc2012 import VOC2012
from datasets.val_transforms import ResizedBaseTransform
from env import VOC2012_DIR


def build():
    return VOC2012(VOC2012_DIR, 'val', ResizedBaseTransform((320, 480), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))


if __name__ == '__main__':
    dataset = build()
    img, target = dataset[2]
    print(img.shape, img.dtype, type(img), (img.min(), img.max()))
    print(target.shape, target.dtype, type(target), (target.min(), target.max()))
    print(len(dataset))
