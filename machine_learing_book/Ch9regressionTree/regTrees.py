from numpy import *


def load_data_set(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = map(float, cur_line)
        data_mat.append(flt_line)
    return data_mat


def bin_split_data_set(data_set, feature, value):
    mat0 = data_set[nonzero(data_set[:feature] > value)[0], :][0]
    mat1 = data_set[nonzero(data_set[:feature] <= value)[0], :][0]
    return mat0, mat1


def reg_leaf(data_set):
    return mean(data_set[:, -1])


def reg_err(data_set):
    return var(data_set[:, -1]) * shape(data_set)[0]


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = shape(data_set)
    s = err_type(data_set)
    best_s = inf
    best_index = 0
    best_value = 0
    for feat_index in range(n - 1):
        for split_val in set(data_set[:, feat_index]):
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val)
            if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    if s - best_s < tol_s:
        return None, leaf_type(data_set)
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)
    if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
        return None, leaf_type(data_set)
    return best_index, best_value


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)
    if feat is None:
        return val
    ret_tree = []
    ret_tree['spind'] = feat
    ret_tree['spval'] = val
    l_set, r_set = bin_split_data_set(data_set, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(l_set, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def is_not_tree(obj):
    return not is_tree(obj)


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])

    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    if shape(test_data)[0] == 0:
        return get_mean(tree)
    if is_tree(tree['right']) or is_tree(tree['left']):
        l_set, r_set = bin_split_data_set(test_data, tree['spind'], tree['spval'])

        if is_tree(tree['left']):
            prune(tree['left'], l_set)
        if is_tree(tree['right']):
            prune(tree['right'], r_set)

    if is_not_tree(tree['right']) and is_not_tree(tree['left']):
        l_set, r_set = bin_split_data_set(test_data, tree['spind'], tree['spval'])
        error_no_merge = sum(power(l_set[:-1] - tree['left'], 2)) + sum(power(r_set[:-1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = sum(power(test_data[:-1], tree_mean, 2))
        if error_merge < error_no_merge:
            print("merging")
            return tree_mean
        else:
            return tree
    else:
        return tree


def linear_solve(data_set):
    m, n = shape(data_set)
    x = mat(ones((m, n)))
    y = mat(ones((m, n)))
    x[:, 1:n] = data_set[:, 0:n - 1]
    y = data_set[:, -1]
    xTx = x.T * x
    if linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, cannot do inverse,"
                        "try increasing the second value of ops")
    ws = xTx.I * (x.T * y)
    return ws, x, y


def model_leaf(data_set):
    ws, x, y = linear_solve(data_set)
    y_hat = x * ws
    return sum(power(y - y_hat, 2))


def model_err(data_set):
    ws, x, y = linear_solve(data_set)
    y_hat = x * ws
    return sum(power(y - y_hat, 2))


def reg_tree_eval(model, in_dat):
    return float(model)


def model_tree_eval(model, in_dat):
    n = shape(in_dat)[1]
    x = mat(ones((1, n + 1)))
    x[:, 1:n + 1] = in_dat
    return float(x * model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    if is_not_tree(tree):
        return model_eval(tree, in_data)

    if in_data[tree['spind']] > tree['spval']:
        if is_tree(tree['left']):
            return tree_fore_cast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    m = len(test_data)
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, mat(test_data[i]), model_eval)
    return y_hat
