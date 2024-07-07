try:
    from .__init__ import torch, autocast, GradScaler, Path, time, DataLoader, np
    from .base_lr_schedulers import BatchLRScheduler
except:
    from __init__ import torch, autocast, GradScaler, Path, time, DataLoader, np
    from base_lr_schedulers import BatchLRScheduler


class SemanticSegmentationTrainer:
    def __init__(self, model, dataset, loss_function, optimizer, lr_scheduler, evaluator,
                 bsize, num_workers, shuffle=True, use_autocast=True, val_interval=5,
                 checkpoint=None):
        """
        :param model:
        :param dataset:
        :param loss_function: reduction为'mean'
        :param optimizer:
        :param lr_scheduler:  仅支持lr_schedulers.py中的lr_scheduler，仅支持batch级的
        :param evaluator:
        :param bsize:
        :param num_workers:
        :param shuffle:
        :param use_autocast:
        :param val_interval: 评估间隔，会先评估，然后之后的每第val_interval的整数倍的epoch会评估
        :param checkpoint: 若为目录，则该目录要么存在同时包含每个项检查点，要么不存在
        """

        self.model = model
        self.dataset = dataset
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.data_loader = DataLoader(dataset, bsize, shuffle, num_workers=num_workers)

        # 训练的可选选项
        self.use_autocast = use_autocast
        self.val_interval = val_interval

        # 存档点
        self.checkpoint = Path(checkpoint)
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_mIoUs': [],
        }

    def __train_one_epoch(self, device='cuda', verbose=1):
        all_t = time.time()
        if verbose == 1:
            print(
                '----------------------------------------------------------------------------------------------------')

        # 将model和optimizer放到目标设备
        self.model.to(device).train()
        for buffer_dict in self.optimizer.state.values():
            for k, v in buffer_dict.items():
                buffer_dict[k] = v.to(device)

        # 混合精度训练
        if self.use_autocast:
            scaler = GradScaler()

        if verbose == 1:
            print('training prepare complete ... ...')
            print('training start ... ...')

        total_loss = 0  # 整个训练集上每个像素的损失累计和
        total_pixels_num = 0
        batch_start_time = time.time()
        for batch_iter, batch in enumerate(self.data_loader):

            # 加载一个batch
            imgs = batch[0].to(device)
            targets = batch[1].to(device)

            # 训练一个batch
            self.optimizer.zero_grad()
            if self.use_autocast:  # 使用混合精度训练
                with autocast():
                    outputs = self.model(imgs)
                    loss = self.loss_function(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(imgs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()

            if verbose == 1:
                print('batch_iter:{} mean_pixels_loss:{:.4f} lr:{:.4e} batch_time:{:.2f}s'.format(batch_iter,
                                                                                                  loss.item(),
                                                                                                  self.optimizer.param_groups[
                                                                                                      0]['lr'],
                                                                                                  # 只看第一个参数组的lr
                                                                                                  time.time() - batch_start_time))

            # batch级的lr调度
            if isinstance(self.lr_scheduler, BatchLRScheduler):
                self.lr_scheduler.step()

            # 累积总损失, 总的像素数
            batch_pixels_num = (targets != self.dataset.border_index).sum()
            total_loss += loss * batch_pixels_num
            total_pixels_num += batch_pixels_num

            batch_start_time = time.time()

        train_loss = (total_loss / total_pixels_num).item()
        print('train_loss:{:.4f} train_time:{:.2f}min'.format(train_loss, (time.time() - all_t) / 60))
        print('----------------------------------------------------------------------------------------------------')
        return train_loss

    def start(self, epoch, device='cuda', verbose=1):
        if verbose == 1:
            print('==================================================================================================')
        all_t = time.time()
        self.lr_scheduler.verbose = verbose

        # 从头训或加载检查点
        if self.checkpoint is not None:
            # 各个结果的路径
            model_state_dict_path = self.checkpoint / 'model.pth'
            optimizer_state_dict_path = self.checkpoint / 'optimizer.pth'
            lr_scheduler_state_dict_path = self.checkpoint / 'lr_scheduler.pth'
            history_path = self.checkpoint / 'history.pth'
            best_path = self.checkpoint / 'best.pth'

            if not self.checkpoint.exists():  # 结果目录不存在
                self.checkpoint.mkdir(parents=True)  # 创建
                if verbose == 1:
                    print('there is no checkpoint, train from scratch, and will save checkpoint......')
            else:  # 结果目录存在，表示检查点存在
                self.model.load_state_dict(torch.load(model_state_dict_path))
                self.optimizer.load_state_dict(torch.load(optimizer_state_dict_path))
                self.lr_scheduler.load_state_dict(torch.load(lr_scheduler_state_dict_path))
                self.history = torch.load(history_path)
                if verbose == 1:
                    print('there is a checkpoint, load it completly, and will train from this checkpoint......')

        for epoch_i in range(epoch):
            if verbose == 1:
                print('training_epoch:{}'.format(len(self.history['train_losses']) + 1))  # 当前的epoch_iter，从1开始

            # train
            train_loss = self.__train_one_epoch(device, verbose)
            torch.cuda.empty_cache()

            # evaluate
            if epoch_i % self.val_interval == 0:
                val_result = self.evaluator.evaluate(self.model, device=device, verbose=verbose)
                torch.cuda.empty_cache()
            else:
                val_result = {
                    'loss': np.nan,
                    'mIoU': np.nan,
                }

            # # lr_schedule 先不要torch官方的scheduler
            # if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.lr_scheduler.step(val_result['loss'])
            # else:
            #     self.lr_scheduler.step()

            # record
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_result['loss'])  # 会有np.nan，但第一个值不为np.nan
            self.history['val_mIoUs'].append(val_result['mIoU'])  # 会有np.nan，但第一个值不为np.nan
            if verbose:
                print('history record complete......')

            # make checkpoint
            if self.checkpoint is not None:
                torch.save(self.model.state_dict(), model_state_dict_path)
                torch.save(self.optimizer.state_dict(), optimizer_state_dict_path)
                torch.save(self.lr_scheduler.state_dict(), lr_scheduler_state_dict_path)
                torch.save(self.history, history_path)
                if self.history['val_mIoUs'][-1] == np.nanmax(self.history['val_mIoUs']):  # np.nanmax可以求带nan的列表的最大值
                    torch.save(self.model.state_dict(), best_path)
                    print('cur_val_mIoU: {:.2f}% is max, and save it......'.format(self.history['val_mIoUs'][-1] * 100))

                if verbose:
                    print('checkpoint save complete, and best mIoU is {:.2f}%......'.format(
                        np.nanmax(self.history['val_mIoUs']) * 100))

        if verbose == 1:
            print('total_time:{:.2f}min'.format((time.time() - all_t) / 60))
            print('==================================================================================================')

        return self.history


if __name__ == '__main__':
    from tools.model_tools import load_model

    model_dir = r'../train_cache/built_models/voc2012_models/r18'
    model = load_model(model_dir)

    from train_cache.built_datasets import voc2012_train

    dataset = voc2012_train.build()

    from train_cache.built_evaluators import voc2012_val_cce

    evaluator = voc2012_val_cce.build_evaluator()

    from train_cache.built_losses import cce

    lf = cce.build_loss(dataset.border_index, 0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    from lr_schedulers import DADLRScheduler

    lr_scheduler = DADLRScheduler(optimizer, 30000, 0.9)
    trainer = SemanticSegmentationTrainer(model, dataset, lf, optimizer, lr_scheduler, evaluator,
                                          256, 1, True, True, 2,
                                          r'../results/tmp')

    trainer.start(3)
