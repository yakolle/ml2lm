from ml2lm.search.cetune.cv_util import *


def avg_tune(avg_metric_func, y, ps, weights, delta_weight=0.01, min_gain=0., max_pace=-1, non_negative=True,
             max_optimization=False, detail=True):
    p_len = len(weights)
    weights = weights.copy()

    last_score = last_best_score = avg_metric_func(y, ps, weights)
    cur_gain = last_score
    while cur_gain > min_gain:
        for i in range(p_len):
            cur_pace = 0
            for direction in [1, -1]:
                while cur_pace < max_pace or max_pace <= 0:
                    w = weights[i] + direction * delta_weight
                    if non_negative and w < 0:
                        break
                    else:
                        weights[i] = w

                    cur_score = avg_metric_func(y, ps, weights)
                    if (cur_score - last_score) * (1.0 if max_optimization else -1.0) <= 0:
                        weights[i] -= direction * delta_weight
                        break
                    last_score = cur_score
                    cur_pace += 1
        cur_gain = (last_score - last_best_score) * (1.0 if max_optimization else -1.0)

        if detail:
            print(f'last_best_score={last_best_score}, cur_best_score={last_score}, cur_gain={cur_gain}')

        last_best_score = last_score

    return weights


def tune(model, data, init_param, param_dic, measure_func=metrics.accuracy_score, cv_repeat_times=1, data_dir=None,
         balance_mode=None, max_optimization=True, mean_std_coeff=(1.0, 1.0), score_min_gain=1e-4, fit_params=None,
         random_state=0, detail=True, kc=None, inlier_indices=None, holdout_data=None, nthread=1, kfold_func=kfold,
         factor_cache=None, warm_probe=False, task_id=None, cv_scores=None, best_steady_trade_off=0, save_model=False,
         non_ordinal_factors=None, end_time=None):
    def get_next_param(learning_model, train_data, param_key, param_val, inlier_ids, h_data):
        param = {param_key: param_val}
        learning_model.set_params(**param)
        return learning_model, train_data, inlier_ids, h_data

    def update_params(params):
        for param_key, param_val in params:
            params = {param_key: param_val}
            model.set_params(**params)

    return tune_factor(model, data, init_param, param_dic, get_next_param, update_params, balance_mode=balance_mode,
                       cv_repeat_times=cv_repeat_times, measure_func=measure_func, mean_std_coeff=mean_std_coeff,
                       max_optimization=max_optimization, score_min_gain=score_min_gain, random_state=random_state,
                       data_dir=data_dir, nthread=nthread, detail=detail, inlier_indices=inlier_indices,
                       holdout_data=holdout_data, fit_params=fit_params, kc=kc, kfold_func=kfold_func,
                       factor_cache=factor_cache, warm_probe=warm_probe, task_id=task_id, cv_scores=cv_scores,
                       best_steady_trade_off=best_steady_trade_off, save_model=save_model,
                       non_ordinal_factors=non_ordinal_factors, end_time=end_time)


def tune_factor(model, data, init_factor, factor_dic, get_next_elements, update_factors, cv_repeat_times=1, kc=None,
                measure_func=metrics.accuracy_score, balance_mode=None, max_optimization=True, score_min_gain=1e-4,
                mean_std_coeff=(1.0, 1.0), data_dir=None, random_state=0, detail=True, inlier_indices=None,
                holdout_data=None, nthread=1, fit_params=None, kfold_func=kfold, factor_cache=None, warm_probe=False,
                task_id=None, cv_scores=None, best_steady_trade_off=0, save_model=False, non_ordinal_factors=None,
                end_time=None):
    def rebuild_factor_dic():
        for fk, fv in best_factors:
            if non_ordinal_factors is None or (non_ordinal_factors is not None and fk not in non_ordinal_factors):
                fvs = factor_dic[fk]
                num_factor_flag = all([type(ele) in [int, float] for ele in fvs])
                if num_factor_flag or obj_factor_build_flag:
                    fvs_size = len(fvs)
                    new_fvs = []
                    idx = fvs.index(fv)
                    if num_factor_flag:
                        if idx - 2 >= 0:
                            new_fvs.append(fvs[idx - 2])
                        if idx - 1 >= 0:
                            new_fvs.append(fvs[idx - 1])
                    else:
                        new_fvs.append('abc123')
                        if idx - 2 >= 0 and fvs[idx - 2] != 'abc123':
                            new_fvs.append(fvs[idx - 2])
                        if idx - 1 >= 0 and fvs[idx - 1] != 'abc123':
                            new_fvs.append(fvs[idx - 1])
                    new_fvs.append(fv)
                    if idx + 1 < fvs_size:
                        new_fvs.append(fvs[idx + 1])
                    if idx + 2 < fvs_size:
                        new_fvs.append(fvs[idx + 2])
                    factor_dic[fk] = new_fvs

    def get_last_best_score():
        _extra_factor_dic = get_valid_function_parameters(get_cur_cv_score, optional_factor_dic)
        cv_score_mean, cv_score_std = get_cur_cv_score(
            model, data, last_best_factors, cur_factor_key, get_next_elements, dict(best_factors),
            random_state=seed_dict[cur_factor_key], **_extra_factor_dic)

        if detail:
            print(f'score of last best factors: mean={cv_score_mean}, std={cv_score_std}')
        return calc_cv_score(cv_score_mean, cv_score_std, mean_std_coeff=mean_std_coeff,
                             max_optimization=max_optimization)

    if data_dir is not None:
        cache_dir = os.path.join(data_dir, 'cache', task_id)
        create_dir(cache_dir)

        tmp_file_name = os.path.join(cache_dir, 'tmp')
        if os.path.exists(tmp_file_name):
            with open(tmp_file_name, 'r') as tmp_file:
                cache_str = tmp_file.readline()
                if cache_str:
                    cv_scores = [float(cv_score) for cv_score in cache_str[:-1].split(',')]
        else:
            open(tmp_file_name, 'w').close()

    optional_factor_dic = {'measure_func': measure_func, 'cv_repeat_times': cv_repeat_times, 'detail': detail,
                           'max_optimization': max_optimization, 'kc': kc, 'inlier_indices': inlier_indices,
                           'mean_std_coeff': mean_std_coeff, 'score_min_gain': score_min_gain, 'data_dir': data_dir,
                           'holdout_data': holdout_data, 'balance_mode': balance_mode, 'nthread': nthread,
                           'fit_params': fit_params, 'kfold_func': kfold_func, 'factor_cache': factor_cache,
                           'task_id': task_id, 'cv_scores': cv_scores, 'save_model': save_model, 'end_time': end_time}

    init_factor_dic = backup(factor_dic)
    best_factors = init_factor
    best_score_pair = None
    seed_dict = {}
    for i, (factor_key, factor_val) in enumerate(best_factors):
        seed_dict[factor_key] = random_state + i
        factor_values = factor_dic[factor_key]
        if factor_val not in factor_values:
            factor_values.append(factor_val)
            factor_dic[factor_key] = sorted(factor_values)
    last_best_factors = backup(best_factors)

    rebuild_dic_flag = warm_probe
    obj_factor_build_flag = warm_probe
    tmp_hold_factors = []
    cur_best_score = 1e10
    cur_factor_key = None
    while True:
        update_factors(best_factors)

        if rebuild_dic_flag:
            rebuild_factor_dic()
        else:
            rebuild_dic_flag = True
            obj_factor_build_flag = True

        for i, (factor_key, factor_val) in enumerate(best_factors):
            if factor_key not in tmp_hold_factors:
                factor_values = factor_dic[factor_key]
                if all([type(ele) in [int, float] for ele in factor_values]):
                    extra_factor_dic = get_valid_function_parameters(probe_best_factor, optional_factor_dic)
                    best_factor_val, mean, std, cur_score_gain, (last_mean, last_std) = probe_best_factor(
                        model, data, factor_key, factor_values, get_next_elements, dict(best_factors),
                        random_state=seed_dict[factor_key], **extra_factor_dic)
                else:
                    extra_factor_dic = get_valid_function_parameters(bootstrap_k_fold_cv_factor, optional_factor_dic)
                    best_factor_val, mean, std, cur_score_gain, (last_mean, last_std) = bootstrap_k_fold_cv_factor(
                        model, data, factor_key, [fv for fv in factor_values if 'abc123' != fv], get_next_elements,
                        dict(best_factors), random_state=seed_dict[factor_key], **extra_factor_dic)

                if cur_score_gain > score_min_gain or not best_steady_trade_off:
                    best_factors[i] = factor_key, best_factor_val
                    best_score_pair = mean, std
                else:
                    best_score_pair = last_mean, last_std
                    if factor_val not in factor_values:
                        factor_values.append(factor_val)
                        factor_dic[factor_key] = sorted(factor_values)
                if cur_score_gain <= score_min_gain:
                    tmp_hold_factors.append(factor_key)

                print('probed factor: ', factor_key, '=', best_factors[i][1], ', mean=', best_score_pair[0], ', std=',
                      best_score_pair[1])
                update_factors([best_factors[i]])
                cur_best_score = calc_cv_score(best_score_pair[0], best_score_pair[1], mean_std_coeff=mean_std_coeff,
                                               max_optimization=max_optimization)
                cur_factor_key = factor_key
        print(best_factors)

        last_best_score = get_last_best_score()
        if (1.0 if max_optimization else -1.0) * (cur_best_score - last_best_score) < score_min_gain \
                or last_best_factors == best_factors:
            print(f'score of final best factors: mean={best_score_pair[0]}, std={best_score_pair[1]}')
            return best_factors, best_score_pair
        if len(tmp_hold_factors) == len(last_best_factors):
            tmp_hold_factors = []
            last_best_factors = backup(best_factors)
            rebuild_dic_flag = False
            obj_factor_build_flag = False
            for factor_key, factor_val in best_factors:
                if not all([type(ele) in [int, float] for ele in factor_dic[factor_key]]):
                    factor_dic[factor_key] = backup(init_factor_dic[factor_key])
