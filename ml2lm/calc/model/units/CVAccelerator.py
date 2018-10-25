import numpy as np
from keras.callbacks import Callback


class CVAccelerator(Callback):
    def __init__(self, monitor='val_loss', tol=1e-4, regret_rate=0.1, cooldown=2, verbose=0, mode='auto'):
        super(CVAccelerator, self).__init__()
        self.monitor = monitor
        self.tol = tol
        self.regret_rate = regret_rate
        self.cooldown = cooldown
        self.cooldown_counter = cooldown
        self.verbose = verbose
        self.mode = mode
        self.best_weights = None

        if self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor):
            self.cmp_op1 = np.less
            self.cmp_op2 = lambda a, b: np.greater(a, b + self.tol)
            self.best = np.Inf
        else:
            self.cmp_op1 = np.greater
            self.cmp_op2 = lambda a, b: np.less(a, b - self.tol)
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if self.cmp_op1(current, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
        elif self.cmp_op2(current, self.best):
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            else:
                if self.verbose > 0:
                    print('Epoch %05d: CVAccelerator triggered' % (epoch + 1))
                cur_weights = self.model.get_weights()
                weights = [(1 - self.regret_rate) * cur_weights[i] + self.regret_rate * self.best_weights[i] for i in
                           range(len(cur_weights))]
                self.model.set_weights(weights)
                self.cooldown_counter = self.cooldown
