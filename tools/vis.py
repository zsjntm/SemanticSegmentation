"""可视化图像，数据的模块"""
try:
    from .__init__ import torch, plt, DataLoader, Path, sys, Image, np, cv2, time, copy, pd
    from .img_process import Tensor2img
except:
    from __init__ import torch, plt, DataLoader, Path, sys, Image, np, cv2, time, copy, pd
    from img_process import Tensor2img


def get_cityscapes_palette():
    palette_foreground = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        70, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]
    palette_padding = [0] * (768 - 19 * 3 - 3)
    palette_border = [0, 0, 0]
    return palette_foreground + palette_padding + palette_border


PALETTE_CITYSCAPES = get_cityscapes_palette()
PALETTE_VOC2012 = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64,
                   0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0,
                   64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64,
                   64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128,
                   0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192,
                   64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128,
                   192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128,
                   192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192,
                   192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128,
                   128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128,
                   128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32,
                   192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128,
                   96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0,
                   192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224,
                   0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192,
                   160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96,
                   64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0,
                   32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0,
                   64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0,
                   0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0,
                   64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160,
                   64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192,
                   160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64,
                   128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224,
                   64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
                   160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0,
                   96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96,
                   0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224,
                   96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64,
                   160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96,
                   32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160,
                   192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192,
                   160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96,
                   224, 192, 224, 224, 192]


def plot_nan_list(lis, linestyle=None, marker=None, color=None, label=None, start_x=1):
    """
    x从start_x开始, 只是将曲线处理后plot
    """
    lis = np.asarray(lis, dtype='float64')

    # 跳过nan值，直接连到下一个非nan的x
    mask = np.isfinite(lis)
    xs = np.arange(start_x, len(mask) + start_x)[mask]
    ys = lis[mask]
    plt.plot(xs, ys, linestyle=linestyle, marker=marker, color=color, label=label)


def vis_history(history_path, show=True):
    """
    能够处理history中的train_losses，val_losses, val_mIoUs中的nan
    """
    history = torch.load(history_path)
    if show == True:
        train_losses, val_losses, val_mIoUs = history['train_losses'], history['val_losses'], history['val_mIoUs']

        plt.title('train and val losses')
        plot_nan_list(train_losses, ' ', 'o', 'b', 'train_loss')
        plot_nan_list(val_losses, None, None, 'b', 'val_loss')
        plt.legend()
        plt.show()

        plt.title('val mIoUs')
        plot_nan_list(val_mIoUs, None, None, 'b', 'val_mIoU')
        plt.legen()
        plt.show()

    return history


def vis_histories(histories, colors=None, labels=None, types=['train_losses', 'val_losses', 'val_mIoUs'], clip_min=None,
                  clip_max=999, start_epoch=1, end_epoch=None):
    """
    show [start_epoch, end_epoch]之间的内容，start_epoch从1开始，end_epoch为None表示show到最后一个
    :param histories: [history1, ...]
    """
    flag = False
    if labels is None:
        labels = range(len(histories))  # 默认为[0, 1, ...]

    if colors is None:
        colors = [None for _ in labels]  # 默认为[None, None, ...]

    for type in types:
        for i, history in enumerate(histories):
            label, color = labels[i], colors[i]

            if end_epoch is None:
                end_epoch = len(history[type])
                flag = True

            lis = history[type][start_epoch - 1: end_epoch]  # 切片
            plot_nan_list(lis, None, None, color, label, start_epoch)

            if flag:
                end_epoch = None
        plt.legend()
        plt.title(type)
        plt.show()


def vis_histories_extremum(histories, start_epoch=1, end_epoch=None):
    """
    表格中的每一项数据都是在[start_epoch, end_epoch]之间计算出来的
    :param histories: {'model1': hisotry1, ...}
    :param start_epoch: 每个history的开始的epoch
    :param end_epoch: 每个historyd结束的epoch，默认为None，即各自的最终epoch
    """
    flag = False
    keys = list(histories.keys())
    results = [[] for _ in keys]
    for i, key in enumerate(keys):
        if end_epoch is None:
            end_epoch = len(histories[key]['train_losses'])
            flag = True
        results[i].append(key)
        results[i].append('%.4f' % np.nanmin(histories[key]['train_losses'][start_epoch - 1: end_epoch]))
        results[i].append('%.4f' % np.nanmin(histories[key]['val_losses'][start_epoch - 1: end_epoch]))
        results[i].append('%.2f' % (np.nanmax(histories[key]['val_mIoUs'][start_epoch - 1: end_epoch]) * 100))
        results[i].append('%.2f' % (histories[key]['val_mIoUs'][start_epoch - 1: end_epoch][-1] * 100))
        results[i].append(len(histories[key]['train_losses'][start_epoch - 1: end_epoch]))
        if flag:
            end_epoch = None
    df = pd.DataFrame(results,
                      columns=['model', 'min_train_loss', 'min_val_loss', 'max_val_mIoU(%)', 'last_val_mIoU(%)',
                               'train_epoch'])
    return df


def vis_seg_map(map, type='output', mapping='voc2012', show=True):
    """
    :param map: torch.tensor
    :param type: option: 'output': (cls_n, h, w), 'target': (1, h, w)
    :param mapping: 采用哪个数据集的类别到颜色的映射, option: 'voc2012', 'cityscapes'
    :return: 色彩掩码图 RGB
    """

    if type == 'output':
        map = map.argmax(dim=-3)  # (h, w)

    if type == 'target':
        map = map.squeeze()  # (h, w)

    if mapping == 'voc2012':
        palette = PALETTE_VOC2012

    if mapping == 'cityscapes':
        palette = PALETTE_CITYSCAPES

    map = map.detach().clone().to('cpu').numpy().astype('uint8')  # (h, w) 'uint8'
    pil = Image.fromarray(map, mode='P')
    pil.putpalette(palette)
    result = np.asarray(pil.convert('RGB'))

    if show:
        plt.imshow(result)
        plt.show()

    return result


@torch.no_grad()
def seg_dataset(model, dataset, results_dir, tensor2img='default', mapping='voc2012', bsize=256, num_workers=0,
                margin=5, device='cuda',
                verbose=1):
    """
    :param model:
    :param dataset: option: 'voc2012_val'
    :param results_dir:
    :param tensor2img: 将数据集的tensor形式的img转换为RGB形式, option: 'default': IN1k的mean, std解标准化, 再*255；或其他函数
    :param mapping: 采用哪个数据集的类别到颜色的映射, option: 'voc2012', 'cityscapes'
    :param bsize:
    :param num_workers:
    :param device:
    :param verbose:
    :return:
    """
    all_t = time.time()
    model.eval().to(device)
    if tensor2img == 'default':
        tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # make results_dir
    results_dir = Path(results_dir)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        if verbose == 1:
            print('dir is not exist, and make it successfully')

    data_loader = DataLoader(dataset, bsize, False, num_workers=num_workers)
    img_idx = 0
    batch_time = time.time()
    for batch_iter, (imgs, targets) in enumerate(data_loader):
        outputs = model(imgs.to(device))

        for img, output, target in zip(imgs, outputs, targets):
            _, h, w = img.shape
            padding = np.zeros((h, margin, 3), dtype='uint8')
            img = tensor2img(img)
            output = vis_seg_map(output, 'output', mapping, False)
            target = vis_seg_map(target, 'target', mapping, False)

            result = np.concatenate([img, padding, output, padding, target], axis=-2)
            cv2.imwrite(str(results_dir / dataset.imgs[img_idx].name), result[:, :, ::-1])
            img_idx += 1
        if verbose == 1:
            print('batch_iter:{} batch_time:{:.2f}s'.format(batch_iter, time.time() - batch_time))
        batch_time = time.time()

    if verbose == 1:
        print('all_time:{:.2f}min'.format((time.time() - all_t) / 60))


def exponential_smoothing(list, smoothing_factor):
    smoothed_list = []
    for i in range(len(list)):
        if i == 0:
            smoothed_list.append(list[i])
        else:
            smoothed_value = smoothing_factor * list[i] + (1 - smoothing_factor) * smoothed_list[i - 1]
            smoothed_list.append(smoothed_value)
    return smoothed_list


def smooth_dictionary(dictionary, smoothing_factor):
    smoothed_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, list):
            smoothed_dictionary[key] = exponential_smoothing(value, smoothing_factor)
        else:
            smoothed_dictionary[key] = value
    return smoothed_dictionary


def copy_and_smooth(d, smooth_factor=0.9, repeat_times=0):
    d = copy.deepcopy(d)
    for _ in range(repeat_times):
        d = smooth_dictionary(d, smooth_factor)
    return d


def vis_batch_imgs(imgs, num_col=5, margin=5, figsize=(15, 15), dpi=300, cmap=None):
    """
        将批量的图片用0像素值焊接起来。
        Args:
            imgs: [img1, img2, ...], 每个img的形状一样，为RGB(h, w, 3)或单通道图np_array(h, w)[0., 1.]
            num_col: 每行展示多少张图片
            margin: 行、列之间的间隔：多少个黑色像素
        Return:
            原imgs用0拼接后的结果
    """

    imgs = np.asarray(imgs)  # (bsize, h, w, [])
    bsize, height, width = imgs.shape[: 3]

    imgs = np.concatenate([imgs, np.zeros((bsize, height, margin) + imgs.shape[3:], dtype='uint8')],
                          axis=2)  # 填充每一列之间的间隔
    imgs = np.concatenate([imgs, np.zeros((bsize, margin, width + margin) + imgs.shape[3:], dtype='uint8')],
                          axis=1)  # 填充每一行之间的间隔

    num_row = np.ceil(bsize / num_col).astype('int32')  # 一共多少行(包括最后一行)

    results = np.zeros((1, (width + margin) * num_col) + imgs.shape[3:],
                       dtype='uint8')  # 结果的上沿 (1, (w + margin) * num_col, [])
    for i in range(num_row):
        tmp = np.concatenate(imgs[i * num_col: (i + 1) * num_col],
                             axis=1)  # 拼接一行的图像，若该行是满的: (h + margin, (w + margin) * num_col, [])

        if tmp.shape[1] < results.shape[1]:  # 若最后一行不是满的，则用0补齐
            tmp = np.concatenate([tmp,
                                  np.zeros((height + margin, results.shape[1] - tmp.shape[1]) + imgs.shape[3:],
                                           dtype='uint8')], axis=1)

        results = np.concatenate([results, tmp], axis=0)  # 和上沿拼起来

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(results, cmap=cmap)
    plt.show()

    return results


def min_max(img, by_channel=False):
    """
        Args:
            img: np_array: (height, width, [channel])
            by_channel: 是否逐通道进行min_max化，默认直接整体进行min_max化。如果by_channel为False，那么img可以是任意维度的np_array。
        Return:
            归一化后的np_array
    """
    if by_channel:
        img = (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1)) + 1e-7)  # 逐通道进行min_max化
    else:
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)  # 整个图片整体进行min_max化
    return img


if __name__ == '__main__':
    "测试代码"
    # img_path = VOC2012_DIR / 'train/label_images/2007_000032.png'
    # palette = Image.open(img_path).getpalette()
    # print(PALETTE_CITYSCAPES)

    a = [1, 2, 3, np.nan, 3, np.nan]
    b = [3, 3, 3]
    # plot_nan_list(a)
    # plot_nan_list(b)
    # plt.show()
    # print(np.nanmin(a), np.nanmax(a))
    # plt.plot(b, color=None, linestyle=None, marker=None, label=None)
    # plt.show()
    print(np.nan * 100)
