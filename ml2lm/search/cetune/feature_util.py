from ml2lm.search.cetune.cv_util import *


def screen_feature(df, var_threshold=.01, corr_threshold=.85, detail=False):
    if detail:
        print(df.shape)
    drop_columns = []
    for col_name, column in df.iteritems():
        if column.dtype in [np.uint8, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
            if column.var() < var_threshold:
                drop_columns += [col_name]
    if detail:
        print(len(drop_columns))

    if corr_threshold < 1:
        columns_dtypes = []
        for col_name, dtype in zip(df.columns, df.dtypes):
            if col_name not in drop_columns and dtype in [np.uint8, np.int8, np.int16, np.int32, np.int64, np.float16,
                                                          np.float32, np.float64]:
                columns_dtypes.append((col_name, dtype))

        size = len(columns_dtypes)
        for i in range(size):
            col_name = columns_dtypes[i][0]
            if col_name not in drop_columns:
                if detail:
                    print('--------', col_name, i, '--------')
                column = df[col_name]
                for j in range(i + 1, size):
                    col_name1 = columns_dtypes[j][0]
                    if col_name1 not in drop_columns:
                        corr_score = column.corr(df[col_name1])
                        if corr_score > corr_threshold:
                            drop_columns += [col_name1]
                            if detail:
                                print(col_name1, corr_score)

    df.drop(drop_columns, axis=1, inplace=True)


def read_cache(model, data_dir, operation_key):
    file_path = data_dir + 'cache\\' + type(model).__name__ + '-' + operation_key
    if os.path.exists(file_path):
        with open(file_path, 'r') as cache_file:
            cache_str = cache_file.readline()
            if cache_str:
                return eval(cache_str)
    return {}


def write_cache(model, cache_map, data_dir, operation_key):
    if cache_map:
        file_path = data_dir + 'cache\\' + type(model).__name__ + '-' + operation_key
        with open(file_path, 'w') as cache_file:
            cache_file.write(str(cache_map))


def greedy_prune_feature(estimator, x, y, min_feature=1, random_state=0, measure_func=metrics.accuracy_score,
                         fit_params=None, balance_mode=None, repeat_times=1, kc=None, inlier_indices=None,
                         holdout_data=None, group_bounds=None, mean_std_coeff=(1.0, 1.0), max_optimization=True,
                         detail=False, nthread=1, prune_cols=None, data_dir=None, operation_key='prune'):
    def cv(train_x):
        cache_key = encode_list(code_map, train_x.columns)
        if cache_key not in score_cache:
            cv_scores = bootstrap_k_fold_cv_train(estimator, (train_x, y), **optional_factor_dic)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            score_cache[cache_key] = cv_mean, cv_std

            if data_dir is not None:
                cur_time = int(datetime.now().timestamp())
                if cur_time - last_time[0] >= 300:
                    last_time[0] = cur_time
                    write_cache(estimator, score_cache, data_dir, operation_key)
                    print('cached')
        else:
            cv_mean = score_cache[cache_key][0]
            cv_std = score_cache[cache_key][1]
        return cv_mean, cv_std

    optional_factor_dic = {'measure_func': measure_func, 'repeat_times': repeat_times, 'holdout_data': holdout_data,
                           'balance_mode': balance_mode, 'nthread': nthread, 'random_state': random_state, 'kc': kc,
                           'inlier_indices': inlier_indices, 'fit_params': fit_params, 'group_bounds': group_bounds}

    if prune_cols is None:
        prune_cols = []
        cur_x = x
    else:
        cur_x = x.drop(prune_cols, axis=1)
        prune_cols = backup(prune_cols)

    if data_dir is not None:
        score_cache = read_cache(estimator, data_dir, operation_key)
    else:
        score_cache = {}
    code_map = dict([(col, i) for i, col in enumerate(x.columns)])

    last_time = [int(datetime.now().timestamp())]
    while len(cur_x.columns) > min_feature:
        cv_score_means = []
        cv_score_stds = []

        ori_cv_mean, ori_cv_std = cv(cur_x)
        cv_score_means.append(ori_cv_mean)
        cv_score_stds.append(ori_cv_std)

        for col in cur_x.columns:
            cv_score_mean, cv_score_std = cv(cur_x.drop(col, axis=1))
            cv_score_means.append(cv_score_mean)
            cv_score_stds.append(cv_score_std)

            if detail:
                print('----------------prune[', col, '], cur[mean=', cv_score_mean, ', std=', cv_score_std,
                      '], diff[mean=', cv_score_mean - ori_cv_mean, ', std=', cv_score_std - ori_cv_std,
                      ']---------------')

        cur_prune_col_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                                    max_optimization=max_optimization)
        if 0 == cur_prune_col_index:
            break
        prune_col = cur_x.columns[cur_prune_col_index - 1]
        cur_x = cur_x.drop(prune_col, axis=1)
        prune_cols.append(prune_col)

        if detail:
            print('----------------last[mean=', cv_score_means[0], ', std=', cv_score_stds[0], '], prune[', prune_col,
                  '], cur[mean=', cv_score_means[cur_prune_col_index], ', std=', cv_score_stds[cur_prune_col_index],
                  ']---------------')
            print(prune_cols)
            print()

    return prune_cols


def greedy_select_feature(estimator, x, y, init_cols=None, max_feature=None, random_state=0, max_optimization=True,
                          measure_func=metrics.accuracy_score, fit_params=None, balance_mode=None, repeat_times=1,
                          kc=None, inlier_indices=None, holdout_data=None, group_bounds=None, mean_std_coeff=(1.0, 1.0),
                          detail=False, nthread=1, data_dir=None, operation_key='select', group_feature_prefixes=None):
    def get_feature_prons(col_names):
        if group_feature_prefixes is not None:
            col_set = set()
            for col_name in col_names:
                col_pron = col_name
                for prefix in group_feature_prefixes:
                    if col_name.startswith(prefix):
                        col_pron = prefix
                        break
                col_set.add(col_pron)
            return sorted(list(col_set))
        else:
            return col_names

    def get_features_by_pron(prefix):
        if group_feature_prefixes is not None:
            return [col_name for col_name in x.columns if col_name.startswith(prefix)]
        else:
            return [prefix]

    def cv():
        cache_key = encode_list(code_map, get_feature_prons(cur_x.columns))
        if cache_key not in score_cache:
            cv_scores = bootstrap_k_fold_cv_train(estimator, (cur_x, y), **optional_factor_dic)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            score_cache[cache_key] = cv_mean, cv_std

            if data_dir is not None:
                cur_time = int(datetime.now().timestamp())
                if cur_time - last_time[0] >= 300:
                    last_time[0] = cur_time
                    write_cache(estimator, score_cache, data_dir, operation_key)
                    print('cached')
        else:
            cv_mean = score_cache[cache_key][0]
            cv_std = score_cache[cache_key][1]
        return cv_mean, cv_std

    optional_factor_dic = {'measure_func': measure_func, 'repeat_times': repeat_times, 'holdout_data': holdout_data,
                           'balance_mode': balance_mode, 'nthread': nthread, 'random_state': random_state, 'kc': kc,
                           'inlier_indices': inlier_indices, 'fit_params': fit_params, 'group_bounds': group_bounds}
    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num

    if init_cols is None:
        select_cols = []
    else:
        select_cols = backup(init_cols)
    if max_feature is None:
        max_feature = len(x.columns)

    if data_dir is not None:
        score_cache = read_cache(estimator, data_dir, operation_key)
    else:
        score_cache = {}
    code_map = dict([(col, i) for i, col in enumerate(get_feature_prons(x.columns))])

    cur_x = x[select_cols]
    x = x.drop(select_cols, axis=1)
    last_time = [int(datetime.now().timestamp())]
    while len(cur_x.columns) < max_feature:
        cv_score_means = []
        cv_score_stds = []

        if 0 == len(cur_x.columns):
            cv_score_means.append(bad_score)
            cv_score_stds.append(large_num)
        else:
            ori_cv_mean, ori_cv_std = cv()
            cv_score_means.append(ori_cv_mean)
            cv_score_stds.append(ori_cv_std)

        cand_cols = get_feature_prons(x.columns)
        for col in cand_cols:
            cols = get_features_by_pron(col)
            cur_x[cols] = x[cols]
            cv_score_mean, cv_score_std = cv()
            cv_score_means.append(cv_score_mean)
            cv_score_stds.append(cv_score_std)
            cur_x = cur_x.drop(cols, axis=1)

            if detail:
                print('----------------select[', col, '], cur[mean=', cv_score_mean, ', std=', cv_score_std,
                      '], diff[mean=', cv_score_mean - cv_score_means[0], ', std=', cv_score_std - cv_score_stds[0],
                      ']---------------')

        cur_select_col_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                                     max_optimization=max_optimization)
        if 0 == cur_select_col_index:
            break
        select_col = cand_cols[cur_select_col_index - 1]
        cur_select_cols = get_features_by_pron(select_col)
        cur_x[cur_select_cols] = x[cur_select_cols]
        x = x.drop(cur_select_cols, axis=1)
        select_cols.append(select_col)

        if detail:
            print('----------------last[mean=', cv_score_means[0], ', std=', cv_score_stds[0], '], select[', select_col,
                  '], cur[mean=', cv_score_means[cur_select_col_index], ', std=', cv_score_stds[cur_select_col_index],
                  ']---------------')
            print(select_cols)
            print()

    return select_cols


def combo_features(estimator, x, y, score_min_gain=1e-4, random_state=0, max_optimization=True, repeat_times=1,
                   measure_func=metrics.accuracy_score, fit_params=None, balance_mode=None, kc=None, detail=False,
                   inlier_indices=None, holdout_data=None, group_bounds=None, mean_std_coeff=(1.0, 1.0), nthread=1,
                   data_dir=None, operation_key='combo', max_combo_features=0.2):
    def cv():
        cks = []
        for combo_col in combo_cols:
            ck = ''
            for ele_col in sorted(combo_col.split('*')):
                ck += str(code_map[ele_col]) + '*'
            cks.append(ck)

        cache_key = str(cks)
        if cache_key not in score_cache:
            cv_scores = bootstrap_k_fold_cv_train(estimator, (x, y), **optional_factor_dic)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            score_cache[cache_key] = cv_mean, cv_std

            if data_dir is not None:
                cur_time = int(datetime.now().timestamp())
                if cur_time - last_time[0] >= 300:
                    last_time[0] = cur_time
                    write_cache(estimator, score_cache, data_dir, operation_key)
                    print('cached')
        else:
            cv_mean = score_cache[cache_key][0]
            cv_std = score_cache[cache_key][1]
        return cv_mean, cv_std

    optional_factor_dic = {'measure_func': measure_func, 'repeat_times': repeat_times, 'holdout_data': holdout_data,
                           'balance_mode': balance_mode, 'nthread': nthread, 'random_state': random_state, 'kc': kc,
                           'inlier_indices': inlier_indices, 'fit_params': fit_params, 'group_bounds': group_bounds}

    if data_dir is not None:
        score_cache = read_cache(estimator, data_dir, operation_key)
    else:
        score_cache = {}
    code_map = dict([(col, i) for i, col in enumerate(x.columns)])

    if isinstance(max_combo_features, float):
        max_combo_features = int(max_combo_features * x.shape[1])

    combo_cols = []
    cand_combo_cols = code_map.keys()
    x = x.copy()

    last_time = [int(datetime.now().timestamp())]
    while cand_combo_cols:
        ori_cv_mean, ori_cv_std = cv()
        cv_score = calc_cv_score(ori_cv_mean, ori_cv_std, mean_std_coeff, max_optimization)

        cv_scores = []
        tmp_cand_combo_cols = []
        for col in cand_combo_cols:
            for c in code_map.keys():
                if c not in col:
                    cand_combo_col = (c + '*' + col) if c < col else (col + '*' + c)
                    if cand_combo_col not in tmp_cand_combo_cols:
                        x[cand_combo_col] = x[col] * x[c]
                        combo_cols.append(cand_combo_col)

                        cv_score_mean, cv_score_std = cv()
                        cur_score = calc_cv_score(cv_score_mean, cv_score_std, mean_std_coeff, max_optimization)
                        if (1.0 if max_optimization else -1.0) * (cur_score - cv_score) >= score_min_gain:
                            tmp_cand_combo_cols.append(cand_combo_col)
                            cv_scores.append(cur_score)

                            if detail:
                                print('----------------combo[', cand_combo_col, '], cur[mean=', cv_score_mean, ', std=',
                                      cv_score_std, '], diff[mean=', cv_score_mean - ori_cv_mean, ', std=',
                                      cv_score_std - ori_cv_std, ']---------------')
                        x = x.drop(cand_combo_col, axis=1)
                        combo_cols = combo_cols[:-1]
        if len(tmp_cand_combo_cols) > max_combo_features:
            tmp_cand_combo_cols = sorted(enumerate(tmp_cand_combo_cols), key=lambda pair: cv_scores[pair[0]],
                                         reverse=max_optimization)[:max_combo_features]
            tmp_cand_combo_cols = [pair[1] for pair in tmp_cand_combo_cols]
        if tmp_cand_combo_cols:
            init_cols = list(x.columns) + tmp_cand_combo_cols[0]
            for cand_combo_col in tmp_cand_combo_cols:
                x[cand_combo_col] = 1
                for col in cand_combo_col.split('*'):
                    x[cand_combo_col] *= x[col]
            tmp_cand_combo_cols = np.setdiff1d(
                greedy_select_feature(estimator, x, y, operation_key=operation_key + '_select' + str(len(combo_cols)),
                                      init_cols=init_cols, detail=detail, max_optimization=max_optimization,
                                      mean_std_coeff=mean_std_coeff, data_dir=data_dir, **optional_factor_dic),
                init_cols)
        cand_combo_cols = list(tmp_cand_combo_cols)
        if detail:
            print('----------------cand_combo_cols----------------')
            print(cand_combo_cols)

        combo_cols += cand_combo_cols
        x = x[list(code_map.keys()) + combo_cols]
    write_cache(estimator, score_cache, data_dir, operation_key)

    return combo_cols


def segment_features(estimator, x, y, segment_func, segment_func_params=None, random_state=0, max_optimization=True,
                     repeat_times=1, measure_func=metrics.accuracy_score, fit_params=None, balance_mode=None, kc=None,
                     detail=False, inlier_indices=None, holdout_data=None, nthread=1, group_bounds=None,
                     mean_std_coeff=(1.0, 1.0), data_dir=None, operation_key='segment', min_values=30):
    optional_factor_dic = {'measure_func': measure_func, 'repeat_times': repeat_times, 'holdout_data': holdout_data,
                           'balance_mode': balance_mode, 'nthread': nthread, 'random_state': random_state, 'kc': kc,
                           'inlier_indices': inlier_indices, 'fit_params': fit_params, 'group_bounds': group_bounds}
    x = x.copy()

    seg_map = {}
    no_seg_cols = []
    target = y.values
    for col_name, col in x.iteritems():
        no_seg_cols.append(col_name)
        if col.dtype in [np.int16, np.int32, np.int64, np.float16, np.float32,
                         np.float64] and col.nunique() >= min_values:
            if segment_func_params is not None:
                section_bounds = segment_func(col.values, target, **segment_func_params)
            else:
                section_bounds = segment_func(col.values, target)
            if section_bounds:
                seg_map[col_name] = section_bounds
                seg_col_name = col_name + '_seg'
                x[seg_col_name] = col
                for i, (l, r) in enumerate(section_bounds):
                    x.loc[(col >= l) & (col < r), seg_col_name] = i
                no_seg_cols = no_seg_cols[:-1]

    cand_seg_cols = np.setdiff1d(
        greedy_select_feature(estimator, x, y, operation_key=operation_key + '_select', init_cols=no_seg_cols,
                              detail=detail, max_optimization=max_optimization, mean_std_coeff=mean_std_coeff,
                              data_dir=data_dir, **optional_factor_dic),
        no_seg_cols)
    final_seg_map = {}
    for col_name in cand_seg_cols:
        if col_name.endswith('_seg'):
            final_seg_map[col_name] = seg_map[col_name]
        else:
            no_seg_cols.append(col_name)

    return no_seg_cols, final_seg_map


def onehot_features(estimator, x, y, random_state=0, max_optimization=True, repeat_times=1, operation_key='onehot',
                    measure_func=metrics.accuracy_score, fit_params=None, balance_mode=None, kc=None, detail=False,
                    inlier_indices=None, holdout_data=None, nthread=1, group_bounds=None, mean_std_coeff=(1.0, 1.0),
                    data_dir=None, max_values=30):
    optional_factor_dic = {'measure_func': measure_func, 'repeat_times': repeat_times, 'holdout_data': holdout_data,
                           'balance_mode': balance_mode, 'nthread': nthread, 'random_state': random_state, 'kc': kc,
                           'inlier_indices': inlier_indices, 'fit_params': fit_params, 'group_bounds': group_bounds}

    no_onehot_cols = []
    onehot_prefixes = []
    for col_name in x.columns:
        if x[col_name].nunique() <= max_values:
            onehot_prefixes.append(col_name + '_')
            x = pd.get_dummies(x, columns=[col_name])
        else:
            no_onehot_cols.append(col_name)

    cand_onehot_cols = np.setdiff1d(
        greedy_select_feature(estimator, x, y, operation_key=operation_key + '_select', init_cols=no_onehot_cols,
                              detail=detail, max_optimization=max_optimization, mean_std_coeff=mean_std_coeff,
                              data_dir=data_dir, group_feature_prefixes=onehot_prefixes, **optional_factor_dic),
        no_onehot_cols)

    return [col[:-1] for col in cand_onehot_cols]
