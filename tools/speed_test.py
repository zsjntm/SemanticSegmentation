try:
    from .__init__ import torch, time
except:
    from __init__ import torch, time


def get_fps(model, test_size=(1024, 2048), repeat_times=15, device='cuda'):
    """

    :param model:
    :param test_size:
    :param repeat_times: 计算fps时，前向传播的次数
    :param device:
    :return:
    """
    model.eval().to(device)
    with torch.no_grad():
        # 先前向传播一次，预热
        input = torch.randn(1, 3, *test_size, device=device)
        logits = model(input)
        torch.cuda.synchronize()  # 确保GPU计算完成

        # 多次前向传播统计fps
        t_start = time.time()
        for _ in range(repeat_times):
            logits = model(input)
            torch.cuda.synchronize()
        spend_time = time.time() - t_start
        fps = repeat_times / spend_time
        return fps


def get_ms(model, test_size=(1024, 2048), repeat_times=15, device='cuda'):
    model.eval().to(device)
    with torch.no_grad():
        # 先前向传播一次，预热
        input = torch.randn(1, 3, *test_size, device=device)
        logits = model(input)
        torch.cuda.synchronize()  # 确保GPU计算完成

        # 多次前向传播统计fps
        t_start = time.time()
        for _ in range(repeat_times):
            logits = model(input)
            torch.cuda.synchronize()
        spend_time = (time.time() - t_start) * 1000
        ms = spend_time / repeat_times
        return ms

if __name__ == '__main__':
    # import torchvision
    #
    # model = torchvision.models.resnet18()
    # print('resnet18 fps: {}'.format(get_fps(model)))
    #
    # print('resnet50 fps: {}'.format(get_fps(torchvision.models.resnet50())))
    # print('resnet101 fps:{}'.format(get_fps(torchvision.models.resnet101())))
    # print('convnext-t fps:{}'.format(get_fps(torchvision.models.convnext_tiny())))
    # print('efficientnet-b0 fps:{}'.format(get_fps(torchvision.models.efficientnet_b0())))

    from tools.model_tools import load_model
    pid = load_model(r'../train_cache/built_models/cityscapes_models/classic/pidnet')
    print('pidnet    fps:{:.2f} latency:{:.2f}ms'.format(get_fps(pid), get_ms(pid)))

    r18 = load_model(r'../train_cache/built_models/cityscapes_models/r18')
    print('r18       fps:{:.2f} latency:{:.2f}ms'.format(get_fps(r18), get_ms(r18)))

    es = load_model(r'../train_cache/built_models/cityscapes_models/classic/esnet')
    print('es        fps:{:.2f} latency:{:.2f}ms'.format(get_fps(es), get_ms(es)))

    es_ddr = load_model(r'../train_cache/built_models/voc2012_models/esnet_upgrade/esnet_ddr_revise3')
    print('es_ddr    fps:{:.2f} latency:{:.2f}ms'.format(get_fps(es_ddr), get_ms(es_ddr)))

    es_u = load_model(r'../train_cache/built_models/voc2012_models/esnet_upgrade/esnet_u_revise1')
    print('ed_u      fps:{:.2f} latency:{:.2f}ms'.format(get_fps(es_u), get_ms(es_u)))

    es_u_ddr_revise2 = load_model(r'../train_cache/built_models/voc2012_models/esnet_upgrade/esnet_u+ddr_revise2')
    print('edu_ddr2  fps:{:.2f} latency:{:.2f}ms'.format(get_fps(es_u_ddr_revise2), get_ms(es_u_ddr_revise2)))