try:
    from .__init__ import torch, DataLoader, time
except:
    from __init__ import torch, DataLoader, time


def get_feature(model, x, module):
    """
    利用hook函数，获取当输入为x时，model的某个模块module的输出
    :param model:
    :param x: 一个tensor，能直接输入模型
    :param module: model的某个子模块
    :retrun: feature (b, ...)
    """
    features = []

    def hook(module, _, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(x)
    handle.remove()  # 移除hook函数
    return features[0]


def get_erf(model, module, c_i, y_i, x_i, dataset, bsize, nw, device='cuda', verbose=1):
    all_t = time.time()
    model.eval().to(device)

    num_imgs = 0
    erf_map = 0
    dataloader = DataLoader(dataset, batch_size=bsize, num_workers=nw)
    if verbose == 1:
        print('compute erf start ......')
        print('batch num:{}'.format(len(dataloader)))
    t = time.time()
    for batch_iter, (imgs, targets) in enumerate(dataloader):
        # 给输入的requires_grad设置为True
        imgs = imgs.to(device)
        imgs.requires_grad = True
        num_imgs += imgs.shape[0]

        # 获取中间层的特征，将对应的像素反向传播
        feature = get_feature(model, imgs, module)  # (bsize, c, h, w)
        pixels = feature[:, c_i, y_i, x_i]
        pixels.sum().backward()

        # 将梯度累积
        erf_map = torch.abs(imgs.grad).sum(dim=(0, 1)) + erf_map
        if verbose == 1:
            print('batch_iter:{} batch_time:{:.2f}s'.format(batch_iter, time.time() - t))
        t = time.time()
    erf_map = erf_map / num_imgs
    if verbose == 1:
        print('all spent time:{:.2f}min'.format((time.time() - all_t) / 60))
    return erf_map


if __name__ == '__main__':
    from model_tools import load_model

    model = load_model(r'../train_cache/built_models/voc2012_models/r18',
                       r'../results/voc2012/r18_large_bsize_hyper_search/lr=0.01/model.pth',
                       'cpu')
    # print(model)

    from train_cache.built_datasets import voc2012_val

    dataset = voc2012_val.build()

    "get_feature"
    # img, target = dataset[0]
    # features = get_feature(model, img.unsqueeze(0), model.backbone[5])
    # print(features.shape)
    # features = features.detach().clone().to('cpu').numpy()[0].transpose(1, 2, 0)
    #
    # from tools.vis import min_max, vis_batch_imgs
    # features = min_max(features, True).transpose(2, 0, 1)
    # vis_batch_imgs(features, 8)
    # print(features.shape)

    "get_erf"
    # device = 'cuda'
    # erf = get_erf(model, model.seg_head, 0, 5, 7, dataset, 256, 2, device)
    # import matplotlib.pyplot as plt
    # plt.imshow(erf.detach().clone().to('cpu').numpy())
    # plt.show()
