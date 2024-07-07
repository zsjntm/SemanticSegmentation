try:
    from .base_lr_schedulers import BatchLRScheduler
except:
    from base_lr_schedulers import BatchLRScheduler

class DADLRScheduler(BatchLRScheduler):
    """语义分割任务中使用的经典调度器"""
    def __init__(self, optimizer, max_iter, power):
        super().__init__(optimizer)

        # 变化函数需要参数
        self.max_iter = max_iter
        self.power = power

    def schedule_lr(self):
        factor = (1 - self.cur_iter / self.max_iter) ** self.power  # factor计算公式

        for i, initial_lr in enumerate(self.initial_lrs):
            self.optimizer.param_groups[i]['lr'] = initial_lr * factor  # 直接修改optimizer中的lr


class NOLRScheduler(BatchLRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def schedule_lr(self):
        pass
