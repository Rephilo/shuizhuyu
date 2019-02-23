import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier


def img2vector(file_name):
    ret_mat = np.zeros([1024], int)
    fr = open(file_name)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            ret_mat[i * 32] = lines[i][j]

    return ret_mat


def read_dataset(path):
    file_list = listdir(path)
    num_files = len(file_list)
    data_set = np.zeros([num_files, 1024], int)
    hw_labels = np.zeros([num_files, 10])
    for i in range(num_files):
        file_path = file_list[i]
        digit = int(file_path.split('_')[0])
        hw_labels[i][digit] = 1.0
        data_set[i] = img2vector(path + '/' + file_path)
    return data_set, hw_labels


if __name__ == '__main__':
    train_dataset, train_hwlabels = read_dataset('trainingDigits')
    clf = MLPClassifier(hidden_layer_sizes=(100,),
                        activation='logistic',
                        solver='adam',
                        learning_rate_init=0.0001,
                        max_iter=2000)
    clf.fit(train_dataset, train_hwlabels)

    dataset, hwlabels = read_dataset('testDigits')
    res = clf.predict(dataset)
    error_num = 0
    num = len(dataset)
    for i in range(num):
        if np.sum(res[i] == hwlabels[i]) < 10:
            error_num += 1
    print("total num:", num, " wrong num:", error_num, " wrong rate:", error_num / float(num))
