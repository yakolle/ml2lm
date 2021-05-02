from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def balance(x, y, mode=None, ratio=1.0):
    if mode is not None:
        pos = y[1 == y]
        neg = y[0 == y]
        pos_len = len(pos)
        neg_len = len(neg)
        expect_pos_len = int(neg_len * ratio)
        if pos_len < expect_pos_len:
            if "under" == mode:
                expect_neg_len = int(pos_len / ratio)
                y = pos.append(neg.sample(n=expect_neg_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
            else:
                y = y.append(pos.sample(expect_pos_len - pos_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
        elif pos_len > expect_pos_len:
            if "under" == mode:
                y = neg.append(pos.sample(n=expect_pos_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
            else:
                expect_neg_len = int(pos_len / ratio)
                y = y.append(neg.sample(expect_neg_len - neg_len))
                y = y.sample(frac=1.0)
                x = x.loc[y.index]
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    return x, y


def fill_outlier_by_gaussian(col, sigmath=3.0):
    center = col.median()
    std = col.std()
    col.loc[col < center - sigmath * std] = center - sigmath * std
    col.loc[col > center + sigmath * std] = center + sigmath * std


def fill_outlier_by_iqr(col, iqr_time=1.5):
    q1 = col.quantile(.25)
    q3 = col.quantile(.75)
    span = (q3 - q1) * iqr_time
    col.loc[col < q1 - span] = q1 - span
    col.loc[col > q3 + span] = q3 + span


def fill_outlier_by_percentile(col, percentile=.0027, frac_tol=.0003, method='gauss', max_iteration=20, detail=False):
    def gauss_bound(width):
        center = col.median()
        std = col.std()
        return col.loc[(col < center - width * std) | (col > center + width * std)].shape[0]

    def iqr_bound(width):
        q1 = col.quantile(.25)
        q3 = col.quantile(.75)
        span = (q3 - q1) * width
        return col.loc[(col < q1 - span) | (col > q3 + span)].shape[0]

    bound_method = gauss_bound if 'gauss' == method else iqr_bound
    num_threshold = int(col.shape[0] * percentile)
    num_tol = int(col.shape[0] * frac_tol) + 1

    if num_threshold > 1:
        left_width = 2
        while bound_method(left_width) < num_threshold:
            left_width /= 2
        right_width = 4
        while bound_method(right_width) > num_threshold:
            right_width *= 1.5

        i = 0
        best_width = (left_width + right_width) / 2
        best_span = abs(bound_method(best_width) - num_threshold)
        while i < max_iteration and best_span > num_tol:
            cur_width = (left_width + right_width) / 2
            cur_span = bound_method(cur_width) - num_threshold

            if cur_span <= 0:
                right_width = cur_width
                if -cur_span <= best_span:
                    best_span = -cur_span
                    best_width = cur_width
            else:
                left_width = cur_width

            if detail:
                print(cur_width, left_width, right_width, cur_span, best_span, num_tol, num_threshold, col.shape[0])
            i += 1

        print('-------------best width(', best_width, '), best span(', best_span, ')-------------')

        if 'gauss' == method:
            fill_outlier_by_gaussian(col, sigmath=best_width)
        else:
            fill_outlier_by_iqr(col, iqr_time=best_width)


def get_rows_by_indices(x, indices):
    if hasattr(x, 'iloc'):
        return x.iloc[indices]
    else:
        return x[indices]


def get_rows_by_condition(x, cond_vals):
    if hasattr(x, 'loc'):
        return x.loc[cond_vals]
    else:
        return x[cond_vals]


def set_rows_by_condition(x, cond_vals, vals):
    if hasattr(x, 'loc'):
        x.loc[cond_vals] = vals
    else:
        x[cond_vals] = vals


def get_groups(y, group_bounds):
    if group_bounds is not None:
        groups = y.copy()
        set_rows_by_condition(groups, y < group_bounds[0][0], 0)
        for i, (l, r) in enumerate(group_bounds):
            set_rows_by_condition(groups, (y >= l) & (y < r), i + 1)
        set_rows_by_condition(groups, y >= group_bounds[-1][1], len(group_bounds) + 1)
    else:
        groups = None

    return groups


def insample_outsample_split(x, y, train_size=.5, holdout_num=5, holdout_frac=.7, random_state=0, full_holdout=False,
                             group_bounds=None):
    if isinstance(train_size, float):
        train_size = int(train_size * len(y))
    groups = get_groups(y, group_bounds)
    if groups is None:
        train_index, h_index = ShuffleSplit(n_splits=1, train_size=train_size, test_size=None,
                                            random_state=random_state).split(y).__next__()
    else:
        train_index, h_index = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=None,
                                                      random_state=random_state).split(y, groups).__next__()
    train_x = x.take(train_index)
    train_y = y.take(train_index)
    h_x = x.take(h_index)
    h_y = y.take(h_index)

    groups = get_groups(h_y, group_bounds)
    h_set = []
    for i in range(holdout_num):
        if groups is None:
            off_index, v_index = ShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                              random_state=random_state + i).split(h_y).__next__()
        else:
            off_index, v_index = StratifiedShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                                        random_state=random_state + i).split(h_y, groups).__next__()
        valid_x = h_x.take(v_index)
        valid_y = h_y.take(v_index)
        h_set.append((valid_x, valid_y))

    if full_holdout:
        return train_x, train_y, h_set, h_x, h_y
    return train_x, train_y, h_set
