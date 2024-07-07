from datasets.cityscapes import Cityscapes
from datasets.train_transforms import TrainTransformV2
from env import CITYSCAPES_DIR


def build():
    return Cityscapes(CITYSCAPES_DIR, 'train',
                      TrainTransformV2((1024, 1024), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (0.125, 2),
                                       (1 / 2, 2), 10, 0.5))


if __name__ == '__main__':
    dataset = build()
    print(len(dataset))
    img, target = dataset[0]
    print(img.shape, img.dtype)
    print(target.shape, target.dtype)

    from tools.vis import vis_seg_map
    import matplotlib.pyplot as plt
    from tools.img_process import Tensor2img

    tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = tensor2img(img)
    plt.imshow(img)
    plt.show()
    vis_seg_map(target, 'target', 'cityscapes', True)
