import warnings

import numpy as np
from keras import backend as bk
from keras.callbacks import Callback, ModelCheckpoint


def calc_score(logs, monitor='loss', monitor_op=np.less):
    score = None
    t_score = logs.get(monitor)
    v_score = logs.get(f'val_{monitor}')
    if t_score is not None and v_score is not None:
        score = v_score ** 2 + (1 if monitor_op == np.less else -1) * (t_score - v_score) ** 2
    return score


class CheckpointDecorator(ModelCheckpoint):
    def __init__(self, filepath, monitor='loss', calc_score_func=calc_score, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        super(CheckpointDecorator, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode,
                                                  period)
        self.calc_score_func = calc_score_func

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = self.calc_score_func(logs)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class LRAnnealingByLoss(Callback):
    def __init__(self, loss_lr_pairs, tol=0.0, verbose=0):
        super(LRAnnealingByLoss, self).__init__()
        self.loss_lr_pairs = loss_lr_pairs[::-1]
        self.tol = tol
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        lr = self.loss_lr_pairs.pop()[1]
        bk.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        cur_loss = logs.get('loss')
        if cur_loss is not None and self.loss_lr_pairs:
            loss, lr = self.loss_lr_pairs[-1]
            if cur_loss - loss <= self.tol:
                bk.set_value(self.model.optimizer.lr, lr)
                self.loss_lr_pairs.pop()
                if self.verbose > 0:
                    print('\nEpoch %05d: LRAnnealingByLoss setting learning '
                          'rate to %s.' % (epoch + 1, lr))
