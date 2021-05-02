import copy
import gc
import time
import warnings

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Embedding, Dense
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

from ml2lm.calc.model.units.embed import *
from ml2lm.calc.model.units.rel import *


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
        if monitor_op != np.less:
            t_score = 1. - t_score
            v_score = 1. - v_score
        score = t_score ** 2 + v_score ** 2
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
                    warnings.warn(f'Can save best model only with {self.monitor} and val_{self.monitor} available, '
                                  'skipping.', RuntimeWarning)
                else:
                    if current < self.best:
                        if self.verbose > 0:
                            print('\nEpoch %05d: score improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: score did not improve from %0.5f' %
                                  (epoch + 1, self.best))
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


def make_evaluate_handler(metric_handler, metric_bias=0., metric_scale=1.):
    def _evaluate_handler(ys, ps):
        scores = [metric_scale * (metric_handler(_y, _p) + metric_bias) for _y, _p in zip(ys, ps)]
        n = len(scores)
        if 1 == n:
            return scores[-1]
        return np.sum(np.square(scores))

    return _evaluate_handler


def sample_ev_data(ev_data, data_size=1 / 2, data_split_seed=0):
    cur_ev_data = ev_data
    if data_size is not None and 0. < data_size < 1.:
        cur_ev_data = []
        for _x, _y in ev_data:
            aind, bind = next(ShuffleSplit(n_splits=1, test_size=data_size, random_state=data_split_seed).split(_y))
            if isinstance(_x, dict):
                cur_ev_data.append(({col: val[bind] for col, val in _x.items()}, _y[bind]))
            else:
                cur_ev_data.append((_x[bind], _y[bind]))
    return cur_ev_data


def get_reset_classes(reset_classes=None):
    if reset_classes is not None:
        if isinstance(reset_classes, list):
            reset_classes = tuple([Embedding, SegEmbedding, FMLayer, BiCrossLayer, Dense] + reset_classes)
        elif not isinstance(reset_classes, tuple):
            reset_classes = (Embedding, SegEmbedding, FMLayer, BiCrossLayer, Dense, reset_classes)
    else:
        reset_classes = (Embedding, SegEmbedding, FMLayer, BiCrossLayer, Dense)
    return reset_classes


def group_weight_ids(weights, inds):
    ws = []
    ind_dict = {}
    i, ei, j = 0, len(inds), 0
    while i < ei:
        k = inds[i][0]
        if k not in ind_dict:
            ws.append(weights[k])
            ind_dict[k] = j
            j += 1
        i += 1
    return ws, ind_dict


def weight_reset(model, ev_data, beam_num=5, replace_with=0., replace_axis=None, min_gain=1e-4, data_size=1 / 10,
                 evaluate_handler=make_evaluate_handler(metrics.roc_auc_score, metric_bias=-1., metric_scale=-1.),
                 skip_gain_threshold=0., data_split_seed=0, reset_classes=None, detail=True, pred_batch_size=1024):
    reset_classes = get_reset_classes(reset_classes)
    weights = []
    for layer in model.layers:
        if isinstance(layer, reset_classes):
            weights += [_w for _w in layer.trainable_weights if 2 == len(_w.shape) and _w.shape[-1] > 1]
    values = bk.batch_get_value(weights)

    replace_values = []
    for v in values:
        replace_values.append(replace_with(v, axis=replace_axis) if callable(replace_with) else replace_with)

    def _update_weights(_inds, set_value=True, restore_value=False):
        _ws, _ind_dict = group_weight_ids(weights, _inds)
        _vs = bk.batch_get_value(_ws)
        _vs_bak = copy.deepcopy(_vs)

        for _i, _j in _inds:
            _vs[_ind_dict[_i]][:, _j] = (replace_values[_i][_j] if 0 == replace_axis else replace_values[_i]
                                         ) if not restore_value else values[_i][:, _j]

        if set_value:
            bk.batch_set_value(list(zip(_ws, _vs)))
            return _ws, _vs_bak
        else:
            return _ws, _vs

    def _weight_reset():
        cur_seed = data_split_seed + cur_round if data_split_seed is not None else None
        cur_ev_data = sample_ev_data(ev_data, data_size=data_size, data_split_seed=cur_seed)

        def _calc_score():
            return evaluate_handler([_y for _x, _y in cur_ev_data], [np.squeeze(model.predict(
                _x, batch_size=pred_batch_size)) for _x, _y in cur_ev_data])

        score = _calc_score()

        def _select_weight(_inds, _last_score):
            for _i, _w in enumerate(weights):
                for _j in range(_w.shape[-1]):
                    _cur_ind = (_i, _j)
                    if _cur_ind not in _inds and _cur_ind not in skip_inds:
                        _cur_ws, _cur_vs_bak = _update_weights([_cur_ind])
                        _cur_score = _calc_score()
                        _cur_gain = _last_score - _cur_score
                        _first_gain = _cur_gain
                        if _cur_ind not in gain_dict:
                            gain_dict[_cur_ind] = _first_gain
                        else:
                            _first_gain = gain_dict[_cur_ind]
                        if _cur_gain < skip_gain_threshold:
                            skip_inds.add(_cur_ind)
                        if _cur_gain >= min_gain:
                            cur_weights.append(_cur_ind)
                            if detail:
                                print(f'select: last_inds={_inds}, last_score={_last_score}, added_ind={_cur_ind},',
                                      f'first_gain={_first_gain}, cur_score={_cur_score}, gain={_cur_gain}')
                        bk.batch_set_value(list(zip(_cur_ws, _cur_vs_bak)))
                        del _cur_ws, _cur_vs_bak
                        gc.collect()

        def _prune_weight(_inds, _last_score):
            for _cur_ind in _inds:
                _cur_ws, _cur_vs_bak = _update_weights([_cur_ind], restore_value=True)
                _cur_score = _calc_score()
                _cur_gain = _last_score - _cur_score
                if _cur_gain >= min_gain:
                    cur_weights.append(_cur_ind)
                    if detail:
                        print(f'prune: last_inds={_inds}, last_score={_last_score}, pruned_ind={_cur_ind},',
                              f'cur_score={_cur_score}, gain={_cur_gain}')
                bk.batch_set_value(list(zip(_cur_ws, _cur_vs_bak)))
                del _cur_ws, _cur_vs_bak
                gc.collect()

        cur_weights, cand_weights = [], []
        cur_score = score
        gain_dict = {}
        skip_inds = set()
        while True:
            cur_weights.clear()
            _select_weight(cand_weights, cur_score)
            if not cur_weights:
                break
            cur_ws, cur_vs_bak = _update_weights(cur_weights)
            del cur_ws, cur_vs_bak
            gc.collect()
            cur_score = _calc_score()
            cand_weights += cur_weights
            cur_weights_bak = cur_weights.copy()

            cur_weights.clear()
            _prune_weight(cand_weights, cur_score)
            if not cur_weights:
                break
            cur_ws, cur_vs_bak = _update_weights(cur_weights, restore_value=True)
            del cur_ws, cur_vs_bak
            gc.collect()
            cur_score = _calc_score()
            for ele in cur_weights:
                cand_weights.remove(ele)
            if cur_weights == cur_weights_bak:
                break
        return _update_weights(cand_weights, restore_value=True)

    cur_round = 0
    cand_updates = []
    while cur_round < beam_num:
        if detail:
            print(f'---------------------------round {cur_round}---------------------------')
        cand_updates.append(_weight_reset())
        cur_round += 1
    return cand_updates


class WeightResetter(Callback):
    def __init__(self, ev_data, trigger_lr=1e-4, init_reset_lr=1e-3, reset_threshold=0., reset_times=5,
                 evaluate_handler=make_evaluate_handler(metrics.roc_auc_score, metric_bias=-1., metric_scale=-1.),
                 lr_scheduler=None, pred_batch_size=1024, data_size=1 / 10, data_split_seed=0, reset_classes=None):
        super(WeightResetter, self).__init__()

        self.ev_data = ev_data
        self.trigger_lr = trigger_lr
        self.init_reset_lr = init_reset_lr
        self.reset_threshold = reset_threshold
        self.reset_times = reset_times
        self.evaluate_handler = evaluate_handler
        self.lr_scheduler = lr_scheduler
        self.pred_batch_size = pred_batch_size
        self.data_size = data_size
        self.data_split_seed = data_split_seed
        self.reset_classes = get_reset_classes(reset_classes)

        self.weights = []
        self.resetters = []

        if self.lr_scheduler is not None:
            if not hasattr(self.lr_scheduler, 'reset'):
                if hasattr(self.lr_scheduler, '_reset'):
                    try:
                        self.lr_scheduler.reset = self.lr_scheduler._reset
                    except Exception:
                        self.lr_scheduler.reset = self.lr_scheduler.on_train_begin
                else:
                    self.lr_scheduler.reset = self.lr_scheduler.on_train_begin

        self.reset_lr_decay = (self.trigger_lr / self.init_reset_lr) ** (1 / self.reset_times)
        self.cnt = 0

    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, self.reset_classes):
                for w in layer.trainable_weights:
                    if 2 == len(w.shape) and w.shape[-1] > 1:
                        self.weights.append(w)
                        if isinstance(layer, (Embedding, SegEmbedding)):
                            self.resetters.append(layer.embeddings_initializer)
                        else:
                            self.resetters.append(layer.kernel_initializer)

    def _calc_score(self, ev_data):
        return self.evaluate_handler([y for x, y in ev_data], [np.squeeze(self.model.predict(
            x, batch_size=self.pred_batch_size)) for x, y in ev_data])

    def _update_weights(self, inds, reset=False):
        ws, ind_dict = group_weight_ids(self.weights, inds)
        vs = bk.batch_get_value(ws)
        vs_bak = copy.deepcopy(vs)
        for i, j in inds:
            v = vs[ind_dict[i]]
            v[:, j] = self.resetters[i]((v.shape[0],), dtype=v.dtype) if reset else 0.

        bk.batch_set_value(list(zip(ws, vs)))
        return ws, vs_bak

    def on_epoch_end(self, epoch, logs=None):
        if self.model.optimizer.lr <= self.trigger_lr and self.cnt < self.reset_times:
            cur_seed = (self.data_split_seed + self.cnt) if self.data_split_seed is not None else None
            cur_ev_data = sample_ev_data(self.ev_data, self.data_size, cur_seed)
            score = self._calc_score(cur_ev_data)
            cand_inds = []
            w_cnts = {}
            for i, w in enumerate(self.weights):
                cur_w_cnt = 0
                for j in range(w.shape[-1]):
                    cur_inds = [(i, j)]
                    cur_ws, cur_vs_bak = self._update_weights(cur_inds)
                    cur_gain = score - self._calc_score(cur_ev_data)
                    if cur_gain >= self.reset_threshold:
                        cand_inds += cur_inds
                        cur_w_cnt += 1
                    bk.batch_set_value(list(zip(cur_ws, cur_vs_bak)))
                    del cur_ws, cur_vs_bak
                    gc.collect()
                if cur_w_cnt > 0:
                    w_cnts[w.name] = cur_w_cnt

            if cand_inds:
                self._update_weights(cand_inds, reset=True)
            if self.lr_scheduler is not None:
                self.lr_scheduler.reset()
            last_lr = bk.get_value(self.model.optimizer.lr)
            reset_lr = self.init_reset_lr * self.reset_lr_decay ** self.cnt
            bk.set_value(self.model.optimizer.lr, reset_lr)
            self.cnt += 1
            print(f"WeightResetter has been triggered {self.cnt} times, last_lr={last_lr}, reset_lr={reset_lr}. total",
                  f"{len(cand_inds)} nodes' weight in {len(w_cnts)} kernels resetted. \nKernels' resets: {w_cnts}")
