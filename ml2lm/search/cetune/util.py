import copy
import fnmatch
import inspect
import os
import re
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from pandas import Series


@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


def increase_count(cnt_map, key):
    if key in cnt_map:
        cnt_map[key] += 1
    else:
        cnt_map[key] = 1


def get_time_stamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def backup(obj):
    return copy.deepcopy(obj)


def is_in_function_params(func, param_name):
    return param_name in inspect.signature(func).parameters


def get_valid_function_parameters(func, param_dic):
    all_param_names = list(inspect.signature(func).parameters.keys())

    valid_param_dic = {}
    for param_key, param_val in param_dic.items():
        if param_key in all_param_names:
            valid_param_dic[param_key] = param_val

    return valid_param_dic


def find_file(root_dir, file_pattern, node_types='f'):
    result = []
    for root, dirs, files in os.walk(root_dir):
        nodes = []
        if 'f' in node_types:
            nodes += files
        if 'd' in node_types:
            nodes += dirs

        for name in nodes:
            if fnmatch.fnmatch(name, file_pattern):
                result.append(os.path.join(root, name))
    return result


def write_model(model, model_dir, model_file_name=None, model_id=None):
    model_id = type(model).__name__ if model_id is None else model_id
    model_file_name = f'{get_time_stamp()}_{model_id}' if model_file_name is None else model_file_name
    joblib.dump(model, os.path.join(model_dir, model_file_name))
    return model_file_name


def read_model(model_dir, model_file_name):
    model_file_path = os.path.join(model_dir, model_file_name)
    return joblib.load(model_file_path)


def create_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def batch_predict(model, df, batch_num=1, data_dir=None):
    p = []
    batch_size = int(df.shape[0] / batch_num) + 1
    if data_dir is None:
        for i in range(batch_num):
            x = df.iloc[i * batch_size: (i + 1) * batch_size]
            p = np.append(p, model.predict(x))
    else:
        i = 0
        partial_file_dir = os.path.join(data_dir, 'predict\\partial')
        if os.path.exists(partial_file_dir):
            i = int(max(os.listdir(partial_file_dir), key=lambda file_no: int(file_no)))
        else:
            os.makedirs(partial_file_dir)

        while i < batch_num:
            x = df.iloc[i * batch_size: (i + 1) * batch_size]
            joblib.dump(model.predict(x), os.path.join(partial_file_dir, str(i)))
            i += 1

        file_names = sorted(os.listdir(partial_file_dir), key=lambda file_no: int(file_no))
        for file_name in file_names:
            p = np.append(p, joblib.load(os.path.join(partial_file_dir, file_name)))

        shutil.rmtree(partial_file_dir)

    return p


def arange(start, end, step, ndigits=10):
    arr = []
    ele = round(start, ndigits)
    while ele < end:
        arr.append(ele)
        ele = round(ele + step, ndigits)
    return arr


def encode_list(code_map, li):
    code = 0
    for ele in li:
        code += 2 ** code_map[ele]
    return code


def round_float_str(info):
    def promote(matched):
        return str(float(matched.group()) + 9e-16)

    def trim1(matched):
        return matched.group(1) + matched.group(2)

    def trim2(matched):
        return matched.group(1)

    info = re.sub(r'[\d.]+?9{4,}[\de-]+', promote, info)
    info = re.sub(r'([\d.]*?)\.?0{4,}\d+(e-\d+)', trim1, info)
    info = re.sub(r'([\d.]+?)0{4,}\d+', trim2, info)

    return info


def split_file_path(file_path):
    file_path_pair = file_path.split(".")
    file_name = file_path_pair[0]
    file_name_index = file_name.rindex("\\") + 1
    return file_name[:file_name_index], file_name[file_name_index:], file_path_pair[1]


def get_best_score_index(train_scores, test_scores, train_test_coeff):
    train_scores_s = Series(train_scores)
    train_ranks = train_scores_s.rank(ascending=False)
    test_scores_s = Series(test_scores)
    test_ranks = test_scores_s.rank(ascending=False)
    diff_ranks = (train_scores_s - test_scores_s).abs().rank()
    ranks = (train_ranks * train_ranks).mul(train_test_coeff[0]) + (test_ranks * test_ranks).mul(
        train_test_coeff[1]) + (diff_ranks * diff_ranks).mul(train_test_coeff[2])
    return ranks.idxmin()


def calc_best_score_index_by_rank(means, stds, mean_std_coeff=(1.0, 1.0), max_optimization=True):
    means_s = Series(means)
    if max_optimization:
        mean_ranks = means_s.rank(ascending=False)
    else:
        mean_ranks = means_s.rank(ascending=True)
    stds_s = Series(stds)
    std_ranks = stds_s.rank(ascending=True)
    ranks = (mean_ranks * mean_ranks).mul(mean_std_coeff[0]) + (std_ranks * std_ranks).mul(mean_std_coeff[1])
    return ranks.idxmin()


def calc_best_score_index(means, stds, mean_std_coeff=(1.0, 1.0), max_optimization=True):
    if max_optimization:
        scores = mean_std_coeff[0] * Series(means) - mean_std_coeff[1] * Series(stds)
        return scores.idxmax()
    else:
        scores = mean_std_coeff[0] * Series(means) + mean_std_coeff[1] * Series(stds)
        return scores.idxmin()


def calc_cv_score(mean, std, mean_std_coeff=(1.0, 1.0), max_optimization=True):
    return mean_std_coeff[0] * mean - (1.0 if max_optimization else -1.0) * mean_std_coeff[1] * std


def find_best_param_in_order(measure_func, param_dic, param_key, target, tol=1e-5, max_iteration=30, negative=False,
                             positive=False, detail=False):
    def get_next_left(cur_left: float) -> float:
        return cur_left / 2 if positive else cur_left * 1.5 if negative else cur_left - abs(cur_left) - 1

    def get_next_right(cur_right: float) -> float:
        return cur_right * 1.5 if positive else cur_right / 2 if negative else cur_right + abs(cur_right) + 1

    left_param = get_next_left(param_dic[param_key])
    right_param = get_next_right(param_dic[param_key])
    while True:
        param_dic[param_key] = left_param
        left_score = measure_func(**param_dic) - target
        if abs(left_score) <= tol:
            print('-------------best param(', left_param, '), best span(', left_score, ')-------------')
            return left_param

        param_dic[param_key] = right_param
        right_score = measure_func(**param_dic) - target
        if abs(right_score) <= tol:
            print('-------------best param(', right_param, '), best span(', right_score, ')-------------')
            return right_param

        if left_score * right_score < 0:
            break

        left_param = get_next_left(left_param)
        right_param = get_next_right(right_param)

    i = 0
    best_param = (left_param + right_param) / 2
    param_dic[param_key] = best_param
    best_span = abs(measure_func(**param_dic) - target)
    while i < max_iteration and best_span > tol:
        cur_param = (left_param + right_param) / 2
        param_dic[param_key] = cur_param
        cur_span = measure_func(**param_dic) - target
        param_dic[param_key] = left_param
        left_span = measure_func(**param_dic) - target

        if abs(cur_span) <= best_span:
            best_span = abs(cur_span)
            best_param = cur_param

        if detail:
            print(cur_param, left_param, right_param, cur_span, best_span)

        if cur_span * left_span >= 0:
            left_param = cur_param
        else:
            right_param = cur_param

        i += 1

    print('-------------best param(', best_param, '), best span(', best_span, ')-------------')

    return best_param
