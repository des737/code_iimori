from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class ScheduledOptim(_LRScheduler):
    """ A simple wrapper class for learning rate scheduling
    plotすれば分かるが, これのやりたいこと
    n_warm_stepsまで: 上昇. 0.0010まで上がる.
    その後, 減少し, anneal_stepsに到達したら一気にanneal_rateで減りに来る.
    """

    def __init__(self, warm_up_step, anneal_steps, anneal_rate, max_lr_scale):

        self._optimizer = None
        self.n_warmup_steps = warm_up_step
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self.last_epoch = -1
        self.max_lr = max_lr_scale
        self.max_over = 0

    def _set_optimizer(self, optimizer):
        self._optimizer = optimizer
        super(ScheduledOptim, self).__init__(self._optimizer, self.last_epoch)

    def _get_lr_scale(self):
        """元実装では, self.last_epoch=1の想定.
        一方, pytorch実装に合わせると, 最初の段階ではlast_epoch=0でくる.
        なので, +1して対処.
        更に, max_lrに到達したらそこから減衰が始まるように改造
        """
        lr_1 = np.power(self.last_epoch+1, -0.5)
        lr_2 = np.power(self.n_warmup_steps, -1.5) * (self.last_epoch+1)

        lr = np.min([lr_1, lr_2])
        if lr > self.max_lr:
            self.max_over = 1

        if self.max_over == 1:
            lr = np.power((self.last_epoch+1)/np.power((self.n_warmup_steps**0.5) * self.max_lr, 3), -0.5)

        for s in self.anneal_steps:
            if self.last_epoch+1 > s:
                lr = lr * self.anneal_rate
        return lr

    def get_lr(self):
        lr_scale = self._get_lr_scale()
        return [base_lr * lr_scale for base_lr in self.base_lrs]
