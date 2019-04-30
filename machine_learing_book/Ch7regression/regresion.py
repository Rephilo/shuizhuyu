from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regression(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T * x_mat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xTx = x_mat.T * (weights * x_mat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat):
    """
    计算方差
    :param y_arr:
    :param y_hat:
    :return:
    """
    return ((y_arr - y_hat) ** 2).sum()


def ridge_regression(x_mat, y_mat, lam=2.0):
    xTx = x_mat.T * x_mat
    denom = xTx + eye(shape(x_mat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var
    num_test_pts = 30
    w_mat = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regression(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):
    in_mat = x_mat.copy()
    in_means = mean(in_mat, 0)
    in_var = var(in_mat, 0)
    in_mat = (in_mat - in_means) / in_var
    return in_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)
    m, n = shape(x_mat)
    return_mat = zeros((num_it, n))
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(num_it):
        print(ws.T)
        lowest_error = inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


def test_0():
    x_arr, y_arr = load_data_set('/Users/wangxiao15/Desktop/machinelearninginaction/Ch08/ex0.txt')
    ws = stand_regression(x_arr, y_arr)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    # y_hat = x_mat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()


def test_1():
    x_arr, y_arr = load_data_set('/Users/wangxiao15/Desktop/machinelearninginaction/Ch08/ex0.txt')
    # lwlr(x_arr[0], x_arr, y_arr, 1.0)
    # lwlr(x_arr[0], x_arr, y_arr, 0.01)
    y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.01)
    x_mat = mat(x_arr)
    srtind = x_mat[:, 1].argsort(0)
    x_sort = x_mat[srtind][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[srtind])
    ax.scatter(x_mat[:, 1].flatten().A[0],
               mat(y_arr).T.flatten().A[0],
               s=2,
               c='red')
    plt.show()


def test_2():
    ab_x, ab_y = load_data_set("/Users/wangxiao15/Desktop/machinelearninginaction/Ch08/abalone.txt")
    ridge_weights = ridge_test(ab_x, ab_y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


if __name__ == '__main__':
    test_2()
