import time
import warnings

import numpy as np
from keras import backend as bk
from keras.callbacks import Callback, ModelCheckpoint


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
                current = self.calc_score_func(logs, self.monitor, self.monitor_op)
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


class EVPerEpoch(Callback):
    def __init__(self, oxes, opss, epochs, period=1, pred_batch_size=5000):
        super(EVPerEpoch, self).__init__()
        self.oxes = oxes
        self.opss = opss
        self.epochs = epochs
        self.period = period
        self.pred_batch_size = pred_batch_size

    def on_epoch_end(self, epoch, logs=None):
        if not (epoch + 1) % self.period or self.model.stop_training or epoch + 1 == self.epochs:
            for ox, ops in zip(self.oxes, self.opss):
                op = np.squeeze(self.model.predict(ox, batch_size=self.pred_batch_size))
                ops.append(op)


class TimeMonitor(Callback):
    def __init__(self, early_stopper, end_time, momentum=0.9):
        super(TimeMonitor, self).__init__()
        self.early_stopper = early_stopper
        self.end_time = end_time
        self.momentum = momentum

        self.last_time = time.time()
        self.spend_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            self.early_stopper.stopped_epoch = epoch + 1

        cur_time = time.time()
        cur_spend_time = cur_time - self.last_time
        self.last_time = cur_time
        self.spend_time = self.momentum * max(self.spend_time, cur_spend_time) + (1 - self.momentum) * cur_spend_time
        if cur_time + self.spend_time >= self.end_time:
            self.early_stopper.stopped_epoch = epoch + 1
            self.model.stop_training = True


def make_squash_flip_ratio(flip_ratio, signal_lr, init_lr=1e-3, peak_point=0.25, pressure=2.):
    def _squash(_x):
        return 1 / (1 + abs(_x)) ** pressure

    def _squash_fr(cur_lr):
        lr0 = signal_lr
        lr2 = init_lr
        lr1 = (1 - peak_point) * lr0 + peak_point * lr2
        fr = _squash(lr0 - lr1) if cur_lr < lr1 else _squash(lr2 - lr1)
        return flip_ratio * (_squash(cur_lr - lr1) - fr) / (1 - fr)

    return _squash_fr


def make_constant_flip_ratio(flip_ratio, signal_lr):
    def _fr(cur_lr):
        return flip_ratio if cur_lr >= signal_lr else 0.

    return _fr


class FlipModel(Callback):
    def __init__(self, proto_model, init_flip_ratio=0.1, train_flip_ratio=0., signal_lr=1e-4, axis=-1, need_init=True,
                 train_fr_adjust_func=None):
        super(FlipModel, self).__init__()
        self.proto_model = proto_model
        self.init_flip_ratio = init_flip_ratio
        self.train_flip_ratio = train_flip_ratio
        self.signal_lr = signal_lr
        self.axis = axis
        self.need_init = need_init
        self.train_fr_adjust_func = train_fr_adjust_func

        if self.train_fr_adjust_func is None:
            self.train_fr_adjust_func = make_constant_flip_ratio(self.train_flip_ratio, self.signal_lr)

        self.flipped_weights = self._flip_weights()

    def _flip_weights(self):
        flipped_weights = []
        proto_weights = self.proto_model.trainable_weights
        for pw in proto_weights:
            if self.axis is not None:
                min_w, max_w = bk.min(pw, axis=self.axis, keepdims=True), bk.max(pw, axis=self.axis, keepdims=True)
            else:
                min_w, max_w = bk.min(pw), bk.max(pw)
            flipped_weights.append(min_w + max_w - pw)
        return bk.batch_get_value(flipped_weights)

    def _merge(self, flip_ratio):
        tws = self.model.trainable_weights
        t_weights = bk.batch_get_value(tws)
        ws = []
        for tw, fw in zip(t_weights, self.flipped_weights):
            ws.append((1 - flip_ratio) * tw + flip_ratio * fw)
        bk.batch_set_value(list(zip(tws, ws)))

    def on_train_begin(self, logs=None):
        if self.need_init:
            init_weight = self.proto_model.get_weights()
            self.model.set_weights(init_weight)
            self._merge(self.init_flip_ratio)

    def on_epoch_begin(self, epoch, logs=None):
        if self.train_flip_ratio > 0:
            lr = bk.get_value(self.model.optimizer.lr)
            fr = self.train_fr_adjust_func(lr)
            if fr > 0:
                self._merge(fr)
