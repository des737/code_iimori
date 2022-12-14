import itertools
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR


class hifigan_optim:
    def __init__(
        self,
        learning_rate,
        adam_b1,
        adam_b2
    ) -> None:
        self.learning_rate = learning_rate
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.optim_g = AdamW
        self.optim_d = AdamW

    def _set_model(self, model_dict):
        generator = model_dict['netG']
        mpd = model_dict['netMPD']
        msd = model_dict['netMSD']
        self.optim_g = self.optim_g(generator.parameters(), self.learning_rate, betas=[self.adam_b1, self.adam_b2])
        self.optim_d = self.optim_d(itertools.chain(msd.parameters(), mpd.parameters()),
                                    self.learning_rate, betas=[self.adam_b1, self.adam_b2])

    def state_dict(self):
        return {'g': self.optim_g.state_dict(), 'd': self.optim_d.state_dict()}

    def load_state_dict(self, checkpoint):
        self.optim_g.load_state_dict(checkpoint['g'])
        self.optim_d.load_state_dict(checkpoint['d'])


class hifigan_lr_scheduler:
    def __init__(self, lr_decay) -> None:
        self.lr_decay = lr_decay
        self.scheduler_g = ExponentialLR
        self.scheduler_d = ExponentialLR

    def _set_optimizer(self, optimizer):
        self.scheduler_g = self.scheduler_g(optimizer.optim_g, gamma=self.lr_decay)
        self.scheduler_d = self.scheduler_d(optimizer.optim_d, gamma=self.lr_decay)

    def step(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def state_dict(self):
        return {'g': self.scheduler_g.state_dict(), 'd': self.scheduler_d.state_dict()}

    def load_state_dict(self, checkpoint):
        self.scheduler_g.load_state_dict(checkpoint['g'])
        self.scheduler_d.load_state_dict(checkpoint['d'])

    def get_last_lr(self):
        return self.scheduler_g.get_last_lr()
