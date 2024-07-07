class BatchLRScheduler:
    """提供当前位置，以及历史记录，以及初始学习率"""

    def __init__(self, optimizer):
        self.optimizer = optimizer

        self.cur_iter = 0  # 记录当前batch_iter
        self.lrs_histories = []  # 记录历史lr
        self.initial_lrs = []
        self.verbose = 1

        # 初始化init_lrs, lrs_histories
        for param_group in optimizer.param_groups:
            self.initial_lrs.append(param_group['lr'])
            self.lrs_histories.append([])

    def step(self):  # 放在optimizer.step()后面
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.lrs_histories[i].append(param_group['lr'])  # 添加本batch的lr

        self.cur_iter += 1  # 下个batch的iter
        self.schedule_lr()  # 直接对optimizer的lr进行调整

    def state_dict(self):
        return {
            'cur_iter': self.cur_iter,
            'lrs_histories': self.lrs_histories,
            'initial_lrs': self.initial_lrs,
        }

    def load_state_dict(self, state_dict):
        self.cur_iter = state_dict['cur_iter']
        self.lrs_histories = state_dict['lrs_histories']
        self.initial_lrs = state_dict['initial_lrs']
        if self.verbose == 1:
            print('lr_scheduler load succesfully......')

    def schedule_lr(self):  # 需要子类重写，提供给子类的接口
        """batch_iter:1, 2, ... 的lr变化交给子类负责"""
        pass

