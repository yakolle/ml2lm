import gc
import threading
from collections import Counter

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

from ml2lm.search.cetune.data_util import *
from ml2lm.search.cetune.util import *


def _cv_trainer(learning_model, data, cv_set_iter, measure_func, cv_scores, inlier_indices, balance_mode, lock=None,
                fit_params=None, data_dir=None, task_id=None, model_id=None, detail=False, end_time=None):
    last_time = int(time.time())
    spend_times = []

    x, y = data[0], data[1]
    need_selector = is_in_function_params(measure_func, 'selector')
    local_cv_scores = []
    i = 0
    for tind, vind in cv_set_iter:
        if inlier_indices is not None:
            tind = np.intersect1d(tind, inlier_indices)
        tx = get_rows_by_indices(x, tind)
        ty = get_rows_by_indices(y, tind)

        if hasattr(learning_model, 'warm_start') and learning_model.warm_start:
            model = learning_model
        else:
            model = backup(learning_model)
        if balance_mode is not None:
            fit_x, fit_y = balance(tx, ty, mode=balance_mode)
        else:
            fit_x, fit_y = tx, ty

        if fit_params is None:
            model.fit(fit_x, fit_y)
        else:
            model.fit(fit_x, fit_y, **fit_params)
        del tx, ty, fit_x, fit_y
        gc.collect()

        if model_id is not None:
            write_model(model, os.path.join(data_dir, 'cache', task_id), model_id=f'{model_id}_{i}')

        vx = get_rows_by_indices(x, vind)
        vy = get_rows_by_indices(y, vind)
        vp = model.predict(vx)
        del vx
        gc.collect()

        cv_score = measure_func(vy, vp, get_rows_by_indices(data[2], vind)) if need_selector else measure_func(vy, vp)
        if detail:
            print(cv_score)
        if data_dir is not None:
            if lock is not None:
                lock.acquire()
                cache_cv_score(cv_score, data_dir, task_id)
                lock.release()
            else:
                cache_cv_score(cv_score, data_dir, task_id)
        local_cv_scores.append(cv_score)
        i += 1

        cur_time = int(time.time())
        spend_times.append(cur_time - last_time)
        if end_time is not None and cur_time + 2 * np.max(spend_times) - np.min(spend_times) > end_time:
            exit(0)
        last_time = cur_time

    if lock is None:
        cv_scores += local_cv_scores
    else:
        lock.acquire()
        cv_scores += local_cv_scores
        lock.release()


def kfold(data, n_splits=3, shuffle=True, random_state=0):
    return list(KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(data[0]))


def stratified_kfold(data, n_splits=3, shuffle=True, random_state=0):
    x, y, group_bounds = data
    groups = get_groups(y, group_bounds)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(y, groups))


def group_kfold(data, n_splits=3, shuffle=True, random_state=0):
    x, y, selector = data[0], data[1], data[2]
    col = selector[data[3]].reset_index(drop=True) if len(data) > 3 else pd.Series(selector)
    cnt = sorted(Counter(col).items(), key=lambda pair: (pair[1], pair[0]))

    part_size = len(cnt) // 3
    s1 = np.array([k for k, v in cnt[: part_size]])
    part1 = [(col.loc[col.isin(s1[tind])].index, col.loc[col.isin(s1[vind])].index) for tind, vind in
             KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(s1)]
    s2 = np.array([k for k, v in cnt[part_size: part_size * 2]])
    part2 = [(col.loc[col.isin(s2[tind])].index, col.loc[col.isin(s2[vind])].index) for tind, vind in
             KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(s2)]
    s3 = np.array([k for k, v in cnt[part_size * 2:]])
    part3 = [(col.loc[col.isin(s3[tind])].index, col.loc[col.isin(s3[vind])].index) for tind, vind in
             KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(s3)]

    return [(np.concatenate((tind, part2[i][0], part3[i][0]), axis=None),
             np.concatenate((vind, part2[i][1], part3[i][1]), axis=None)) for i, (tind, vind) in enumerate(part1)]


def bootstrap_k_fold_cv_train(learning_model, data, kfold_func=kfold, statistical_size=30, repeat_times=1, refit=False,
                              random_state=0, measure_func=metrics.accuracy_score, balance_mode=None, kc=None,
                              inlier_indices=None, holdout_data=None, nthread=1, fit_params=None, data_dir=None,
                              task_id=None, cv_scores=None, model_id=None, end_time=None):
    x, y = data[0], data[1]
    if kc is not None:
        k = kc[0]
        c = kc[1]
    else:
        k = int(x.shape[0] / (x.shape[1] * statistical_size))
        if k < 3:
            k = int(x.shape[0] / (statistical_size * 2))
        c = int(np.ceil(statistical_size * repeat_times / k))

    if cv_scores is None:
        cur_cv_scores = []
    else:
        cur_cv_scores = cv_scores.copy()

    cur_scores_num = len(cur_cv_scores)
    c0 = cur_scores_num // k
    k0 = cur_scores_num % k

    if nthread <= 1:
        for i in range(c0, c):
            if random_state is not None:
                random_state += i
            cv_set = kfold_func(data, n_splits=k, shuffle=True, random_state=random_state)[k0:]

            model_name = f'{model_id}_{c0}_{k0}' if model_id is not None else None
            _cv_trainer(learning_model, data, cv_set, measure_func, cur_cv_scores, inlier_indices, balance_mode,
                        fit_params=fit_params, data_dir=data_dir, task_id=task_id, model_id=model_name,
                        end_time=end_time)
    else:
        learning_model = backup(learning_model)
        if hasattr(learning_model, 'warm_start'):
            learning_model.warm_start = False
        lock = threading.RLock()
        cur_cv_scores = []

        for i in range(c):
            if random_state is not None:
                random_state += i
            cv_set = kfold_func(data, n_splits=k, shuffle=True, random_state=random_state)
            batch_size = int(np.ceil(len(cv_set) / nthread))

            tasks = []
            for j in range(nthread):
                cv_set_part = cv_set[j * batch_size: (j + 1) * batch_size]

                model_name = f'{model_id}_{c0}_{k0}_{j}' if model_id is not None else None
                if cv_set_part:
                    t = threading.Thread(target=_cv_trainer, args=(
                        learning_model, data, cv_set_part, measure_func, cur_cv_scores, inlier_indices, balance_mode,
                        lock, fit_params, data_dir, task_id, model_name, False, end_time))
                    tasks.append(t)
            for t in tasks:
                t.start()
            for t in tasks:
                t.join()

    if not cur_cv_scores:
        raise Exception('cur_cv_scores is empty')

    if holdout_data is not None:
        model = backup(learning_model)
        if inlier_indices is not None:
            tr_inlier_indices = np.intersect1d(range(x.shape[0]), inlier_indices)
            x = get_rows_by_indices(x, tr_inlier_indices)
            y = get_rows_by_indices(y, tr_inlier_indices)

        if fit_params is None:
            model.fit(x, y)
        else:
            model.fit(x, y, **fit_params)

        holdout_scores = []
        need_selector = is_in_function_params(measure_func, 'selector')
        for h_data in holdout_data:
            hx, hy = h_data[0], h_data[1]
            hp = model.predict(hx)
            holdout_scores.append(measure_func(hy, hp, h_data[2]) if need_selector else measure_func(hy, hp))

        if refit:
            res = cur_cv_scores, holdout_scores, model
        else:
            res = cur_cv_scores, holdout_scores
    else:
        if refit:
            model = backup(learning_model)
            if inlier_indices is not None:
                tr_inlier_indices = np.intersect1d(range(x.shape[0]), inlier_indices)
                x = get_rows_by_indices(x, tr_inlier_indices)
                y = get_rows_by_indices(y, tr_inlier_indices)

            if fit_params is None:
                model.fit(x, y)
            else:
                model.fit(x, y, **fit_params)
            res = cur_cv_scores, model
        else:
            res = cur_cv_scores

    if data_dir is not None:
        clear_cached_cv_score(data_dir, task_id)
    if cv_scores:
        cv_scores.clear()
    return res


def get_cur_cv_score(model, data, facotrs, factor_key, get_next_elements, factor_table, kfold_func=kfold,
                     cv_repeat_times=1, random_state=0, measure_func=metrics.accuracy_score, nthread=1,
                     balance_mode=None, data_dir=None, kc=None, detail=False, max_optimization=True,
                     inlier_indices=None, holdout_data=None, save_model=False, fit_params=None, factor_cache=None,
                     task_id=None, cv_scores=None, end_time=None):
    if data_dir is not None or factor_cache is not None:
        score_cache = read_cache(model, factor_key, factor_table, data_dir=data_dir, factor_cache=factor_cache,
                                 task_id=task_id)
    else:
        score_cache = {}

    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num

    factor_val = None
    for fk, fv in facotrs:
        model, data, inlier_indices, holdout_data = get_next_elements(model, data, fk, fv, inlier_indices, holdout_data)
        if fk == factor_key:
            factor_val = factor_key

    need_flush = False
    if factor_val not in score_cache:
        try:
            need_flush = True
            model_id = f'{factor_key}_{factor_val}' if save_model else None
            cur_cv_scores = bootstrap_k_fold_cv_train(
                model, data, kfold_func=kfold_func, repeat_times=cv_repeat_times, random_state=random_state,
                measure_func=measure_func, balance_mode=balance_mode, kc=kc, holdout_data=holdout_data,
                inlier_indices=inlier_indices, nthread=nthread, fit_params=fit_params, data_dir=data_dir,
                task_id=task_id, cv_scores=cv_scores, model_id=model_id, end_time=end_time)

            if holdout_data is not None:
                cur_cv_scores, holdout_scores = cur_cv_scores
                cv_score_mean = np.mean(cur_cv_scores)
                cv_score_std = np.std(cur_cv_scores)
                score_cache[factor_val] = cv_score_mean, cv_score_std, holdout_scores
            else:
                cv_score_mean = np.mean(cur_cv_scores)
                cv_score_std = np.std(cur_cv_scores)
                score_cache[factor_val] = cv_score_mean, cv_score_std
        except Exception as e:
            cv_score_mean = bad_score
            cv_score_std = large_num / 10

            print(e)
    else:
        cache_val = score_cache[factor_val]
        if 3 == len(cache_val):
            cv_score_mean, cv_score_std, holdout_scores = cache_val
        else:
            cv_score_mean, cv_score_std = cache_val

    if detail:
        if 'holdout_scores' in dir():
            print('----------------', factor_key, '=', factor_val, ', cv_mean=', cv_score_mean, ', cv_std=',
                  cv_score_std, ', holdout_mean=', np.mean(holdout_scores), ', holdout_std=', np.std(holdout_scores),
                  holdout_scores, '---------------')
        else:
            print('----------------', factor_key, '=', factor_val, ', mean=', cv_score_mean, ', std=', cv_score_std,
                  '---------------')

    if need_flush and (data_dir is not None or factor_cache is not None):
        write_cache(model, factor_key, score_cache, factor_table, data_dir=data_dir, factor_cache=factor_cache,
                    task_id=task_id)
        if factor_cache is not None:
            print(factor_cache)
        else:
            print(get_time_stamp())

    return cv_score_mean, cv_score_std


def bootstrap_k_fold_cv_factor(learning_model, data, factor_key, factor_values, get_next_elements, factor_table,
                               kfold_func=kfold, cv_repeat_times=1, random_state=0, measure_func=metrics.accuracy_score,
                               nthread=1, balance_mode=None, data_dir=None, kc=None, mean_std_coeff=(1.0, 1.0),
                               detail=False, max_optimization=True, inlier_indices=None, holdout_data=None,
                               save_model=False, fit_params=None, factor_cache=None, task_id=None, cv_scores=None,
                               end_time=None):
    if data_dir is not None or factor_cache is not None:
        score_cache = read_cache(learning_model, factor_key, factor_table, data_dir=data_dir, factor_cache=factor_cache,
                                 task_id=task_id)
    else:
        score_cache = {}

    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num

    print(factor_values)
    cv_score_means = []
    cv_score_stds = []
    last_time = int(time.time())
    last_best_factor_score = bad_score
    last_best_factor_score_pair = None
    for factor_val in factor_values:
        need_flush = not cv_scores
        if factor_val not in score_cache:
            try:
                model_id = f'{factor_key}_{factor_val}' if save_model else None
                learning_model, data, inlier_indices, holdout_data = get_next_elements(learning_model, data, factor_key,
                                                                                       factor_val, inlier_indices,
                                                                                       holdout_data)
                cur_cv_scores = bootstrap_k_fold_cv_train(
                    learning_model, data, kfold_func=kfold_func, repeat_times=cv_repeat_times,
                    random_state=random_state, measure_func=measure_func, balance_mode=balance_mode, kc=kc,
                    holdout_data=holdout_data, inlier_indices=inlier_indices, nthread=nthread, fit_params=fit_params,
                    data_dir=data_dir, task_id=task_id, cv_scores=cv_scores, model_id=model_id, end_time=end_time)

                if holdout_data is not None:
                    cur_cv_scores, holdout_scores = cur_cv_scores
                    cv_score_mean = np.mean(cur_cv_scores)
                    cv_score_std = np.std(cur_cv_scores)
                    score_cache[factor_val] = cv_score_mean, cv_score_std, holdout_scores
                else:
                    cv_score_mean = np.mean(cur_cv_scores)
                    cv_score_std = np.std(cur_cv_scores)
                    score_cache[factor_val] = cv_score_mean, cv_score_std
            except Exception as e:
                cv_score_mean = bad_score
                cv_score_std = large_num / 10

                print(e)
        else:
            cache_val = score_cache[factor_val]
            if 3 == len(cache_val):
                cv_score_mean, cv_score_std, holdout_scores = cache_val
            else:
                cv_score_mean, cv_score_std = cache_val
        cv_score_means.append(cv_score_mean)
        cv_score_stds.append(cv_score_std)

        if factor_val == factor_table[factor_key]:
            last_best_factor_score_pair = cv_score_mean, cv_score_std
            last_best_factor_score = calc_cv_score(cv_score_mean, cv_score_std, mean_std_coeff=mean_std_coeff,
                                                   max_optimization=max_optimization)

        if detail:
            if 'holdout_scores' in dir():
                print('----------------', factor_key, '=', factor_val, ', cv_mean=', cv_score_mean, ', cv_std=',
                      cv_score_std, ', holdout_mean=', np.mean(holdout_scores), ', holdout_std=',
                      np.std(holdout_scores), holdout_scores, '---------------')
            else:
                print('----------------', factor_key, '=', factor_val, ', mean=', cv_score_mean, ', std=', cv_score_std,
                      '---------------')

        if data_dir is not None or factor_cache is not None:
            cur_time = int(time.time())
            if cur_time - last_time >= 300 or (need_flush and not cv_scores):
                last_time = cur_time
                write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir,
                            factor_cache=factor_cache, task_id=task_id)
                if factor_cache is not None:
                    print(factor_cache)
                else:
                    print(get_time_stamp())

    if data_dir is not None or factor_cache is not None:
        write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir, factor_cache=factor_cache,
                    task_id=task_id)

    best_factor_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                              max_optimization=max_optimization)
    best_factor = factor_values[best_factor_index]
    print('--best factor: ', factor_key, '=', best_factor, ', mean=', cv_score_means[best_factor_index], ', std=',
          cv_score_stds[best_factor_index])

    cur_best_factor_score = calc_cv_score(cv_score_means[best_factor_index], cv_score_stds[best_factor_index],
                                          mean_std_coeff=mean_std_coeff, max_optimization=max_optimization)

    return best_factor, cv_score_means[best_factor_index], cv_score_stds[best_factor_index], abs(
        cur_best_factor_score - last_best_factor_score), last_best_factor_score_pair


def cache_cv_score(cv_score, data_dir, task_id):
    tmp_score_file_name = os.path.join(data_dir, 'cache', task_id, 'tmp')
    with open(tmp_score_file_name, 'a') as tmp_file:
        tmp_file.write(f'{cv_score},')


def clear_cached_cv_score(data_dir, task_id):
    open(os.path.join(data_dir, 'cache', task_id, 'tmp'), 'w').close()


def get_cache_key(factor_table):
    cache_key = round_float_str(str(sorted(factor_table.items(), key=lambda pair: pair[0])))
    return cache_key


def read_cache(model, factor_key, factor_table, data_dir=None, factor_cache=None, task_id=None):
    if factor_cache is not None:
        return read_cache_from_memory(model, factor_key, factor_table, factor_cache)
    return read_cache_from_file(model, factor_key, data_dir, factor_table, task_id)


def read_cache_from_memory(model, factor_key, factor_table, factor_cache):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    factor_cache_key = f'{type(model).__name__}-{factor_key}'
    if factor_cache_key in factor_cache:
        cache = factor_cache[factor_cache_key]
        cache_key = get_cache_key(factor_table)
        if cache_key in cache:
            return cache[cache_key]

    return {}


def read_cache_from_file(model, factor_key, data_dir, factor_table, task_id):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    file_path = os.path.join(data_dir, 'cache', task_id, f'{type(model).__name__}-{factor_key}')
    if os.path.exists(file_path):
        with open(file_path, 'r') as cache_file:
            cache_str = cache_file.readline()
            if cache_str:
                cache = eval(cache_str)
                cache_key = get_cache_key(factor_table)
                if cache_key in cache:
                    return cache[cache_key]
    return {}


def write_cache(model, factor_key, score_map, factor_table, data_dir=None, factor_cache=None, task_id=None):
    if data_dir is not None:
        write_cache_to_file(model, factor_key, score_map, data_dir, factor_table, task_id)
    if factor_cache is not None:
        write_cache_to_memory(model, factor_key, score_map, factor_table, factor_cache)


def write_cache_to_memory(model, factor_key, score_map, factor_table, factor_cache):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    if score_map:
        cache = {}
        factor_cache_key = f'{type(model).__name__}-{factor_key}'
        if factor_cache_key in factor_cache:
            cache = factor_cache[factor_cache_key]

        cache_key = get_cache_key(factor_table)
        if cache_key in cache:
            cache[cache_key].update(score_map)
        else:
            cache[cache_key] = score_map
        factor_cache[factor_cache_key] = cache


def write_cache_to_file(model, factor_key, score_map, data_dir, factor_table, task_id):
    factor_table = backup(factor_table)
    del factor_table[factor_key]

    if score_map:
        cache = {}
        file_path = os.path.join(data_dir, 'cache', task_id, f'{type(model).__name__}-{factor_key}')
        if os.path.exists(file_path):
            with open(file_path, 'r') as cache_file:
                cache_str = cache_file.readline()
                if cache_str:
                    cache = eval(cache_str)

        with open(file_path, 'w') as cache_file:
            cache_key = get_cache_key(factor_table)
            if cache_key in cache:
                cache[cache_key].update(score_map)
            else:
                cache[cache_key] = score_map
            cache_file.write(round_float_str(str(cache)))


def probe_best_factor(learning_model, data, factor_key, factor_values, get_next_elements, factor_table,
                      kfold_func=kfold, detail=False, cv_repeat_times=1, kc=None, score_min_gain=1e-4,
                      measure_func=metrics.accuracy_score, balance_mode=None, random_state=0, mean_std_coeff=(1.0, 1.0),
                      max_optimization=True, nthread=1, data_dir=None, inlier_indices=None, holdout_data=None,
                      fit_params=None, factor_cache=None, task_id=None, cv_scores=None, save_model=False,
                      end_time=None):
    int_flag = all([isinstance(ele, int) for ele in factor_values])
    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num
    last_best_score = bad_score

    if data_dir is not None or factor_cache is not None:
        score_cache = read_cache(learning_model, factor_key, factor_table, data_dir=data_dir, factor_cache=factor_cache,
                                 task_id=task_id)
    else:
        score_cache = {}

    last_time = int(time.time())
    last_best_factor_score = last_best_score
    last_best_factor_score_pair = None
    while True:
        print(factor_values)
        cv_score_means = []
        cv_score_stds = []
        for factor_val in factor_values:
            need_flush = not cv_scores
            if factor_val not in score_cache:
                try:
                    model_id = f'{factor_key}_{round_float_str(str(factor_val))}' if save_model else None
                    learning_model, data, inlier_indices, holdout_data = get_next_elements(learning_model, data,
                                                                                           factor_key, factor_val,
                                                                                           inlier_indices, holdout_data)
                    cur_cv_scores = bootstrap_k_fold_cv_train(
                        learning_model, data, kfold_func=kfold_func, repeat_times=cv_repeat_times, nthread=nthread,
                        random_state=random_state, measure_func=measure_func, balance_mode=balance_mode, kc=kc,
                        holdout_data=holdout_data, inlier_indices=inlier_indices, fit_params=fit_params,
                        data_dir=data_dir, task_id=task_id, cv_scores=cv_scores, model_id=model_id, end_time=end_time)

                    if holdout_data is not None:
                        cur_cv_scores, holdout_scores = cur_cv_scores
                        cv_score_mean = np.mean(cur_cv_scores)
                        cv_score_std = np.std(cur_cv_scores)
                        score_cache[factor_val] = cv_score_mean, cv_score_std, holdout_scores
                    else:
                        cv_score_mean = np.mean(cur_cv_scores)
                        cv_score_std = np.std(cur_cv_scores)
                        score_cache[factor_val] = cv_score_mean, cv_score_std
                except Exception as e:
                    cv_score_mean = bad_score
                    cv_score_std = large_num / 10

                    print(e)
            else:
                cache_val = score_cache[factor_val]
                if 3 == len(cache_val):
                    cv_score_mean, cv_score_std, holdout_scores = cache_val
                else:
                    cv_score_mean, cv_score_std = cache_val
            cv_score_means.append(cv_score_mean)
            cv_score_stds.append(cv_score_std)

            if factor_val == factor_table[factor_key]:
                last_best_factor_score_pair = cv_score_mean, cv_score_std
                last_best_factor_score = calc_cv_score(cv_score_mean, cv_score_std, mean_std_coeff=mean_std_coeff,
                                                       max_optimization=max_optimization)

            if detail:
                if 'holdout_scores' in dir():
                    print('----------------', factor_key, '=', factor_val, ', cv_mean=', cv_score_mean, ', cv_std=',
                          cv_score_std, ', holdout_mean=', np.mean(holdout_scores), ', holdout_std=',
                          np.std(holdout_scores), holdout_scores, '---------------')
                else:
                    print('----------------', factor_key, '=', factor_val, ', mean=', cv_score_mean, ', std=',
                          cv_score_std, '---------------')

            if data_dir is not None or factor_cache is not None:
                cur_time = int(time.time())
                if cur_time - last_time >= 300 or (need_flush and not cv_scores):
                    last_time = cur_time
                    write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir,
                                factor_cache=factor_cache, task_id=task_id)
                    if factor_cache is not None:
                        print(factor_cache)
                    else:
                        print(get_time_stamp())

        if data_dir is not None or factor_cache is not None:
            write_cache(learning_model, factor_key, score_cache, factor_table, data_dir=data_dir,
                        factor_cache=factor_cache, task_id=task_id)

        scores = calc_cv_score(np.array(cv_score_means), np.array(cv_score_stds), mean_std_coeff=mean_std_coeff,
                               max_optimization=max_optimization)
        best_factor_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                                  max_optimization=max_optimization)
        neigbour_gap = large_num
        if 0 < best_factor_index < len(scores) - 1:
            left_gap = abs(scores[best_factor_index - 1] - scores[best_factor_index])
            right_gap = abs(scores[best_factor_index + 1] - scores[best_factor_index])
            neigbour_gap = (left_gap + right_gap) / 3
            neigbour_gap = max(min(left_gap, right_gap), neigbour_gap)
        if abs(scores[best_factor_index] - last_best_score) < score_min_gain or max(scores) - min(
                scores) < score_min_gain or neigbour_gap < score_min_gain:
            best_factor = factor_values[best_factor_index]
            print('--best factor: ', factor_key, '=', best_factor, ', mean=', cv_score_means[best_factor_index],
                  ', std=', cv_score_stds[best_factor_index])

            cur_best_factor_score = calc_cv_score(cv_score_means[best_factor_index], cv_score_stds[best_factor_index],
                                                  mean_std_coeff=mean_std_coeff, max_optimization=max_optimization)

            return best_factor, cv_score_means[best_factor_index], cv_score_stds[best_factor_index], abs(
                cur_best_factor_score - last_best_factor_score), last_best_factor_score_pair
        last_best_score = scores[best_factor_index]

        factor_size = len(factor_values)
        if max_optimization:
            cur_best_index1 = max(range(factor_size), key=lambda i: cv_score_means[i] + cv_score_stds[i])
            cur_best_index2 = max(range(factor_size), key=lambda i: cv_score_means[i] - cv_score_stds[i])
        else:
            cur_best_index1 = min(range(factor_size), key=lambda i: cv_score_means[i] + cv_score_stds[i])
            cur_best_index2 = min(range(factor_size), key=lambda i: cv_score_means[i] - cv_score_stds[i])

        l = min(cur_best_index1, cur_best_index2) - 1
        r = max(cur_best_index1, cur_best_index2) + 1
        if r >= factor_size:
            r = factor_size
            right_value = factor_values[-1]
            if right_value > 0:
                right_value = round(right_value * 1.5, 10)
            else:
                right_value = round(right_value / 2, 10)
            right_value = int(right_value) if int_flag else right_value
            factor_values.append(right_value)
        if l < 0:
            l = 0
            left_value = factor_values[0]
            if 0 != left_value:
                if left_value > 0:
                    left_value = round(left_value / 2, 10)
                else:
                    left_value = round(left_value * 1.5, 10)
                left_value = int(left_value) if int_flag else left_value
                factor_values.insert(0, left_value)
                r += 1

        step = (factor_values[l + 1] - factor_values[l]) / 2
        step = step if not int_flag else int(np.ceil(step))
        if step <= 1e-10:
            continue

        factor_size = (factor_values[r] - factor_values[l]) / step
        if factor_size < 5:
            step /= 2
        elif factor_size > 16:
            step = (factor_values[r] - factor_values[l]) / 16
        step = step if not int_flag else 1 if step <= 1 else int(step)
        next_factor_values = arange(factor_values[l], factor_values[r] + step, step)
        if factor_values[l + 1] not in next_factor_values:
            next_factor_values.append(factor_values[l + 1])
        if factor_values[r - 1] not in next_factor_values:
            next_factor_values.append(factor_values[r - 1])

        factor_values.clear()
        factor_values.extend(sorted(next_factor_values))
