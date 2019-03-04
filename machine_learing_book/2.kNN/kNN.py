from numpy import *
from os import listdir
import operator


def created_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(int_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(int_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        votel_label = labels[sorted_dist_indicies[i]]
        class_count[votel_label] = class_count.get(votel_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    fr = open(filename)
    array_o_lines = fr.readlines()
    number_of_lines = len(array_o_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_o_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_data_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :],
                                      norm_mat[num_test_vecs:m, :],
                                      dating_data_labels[num_test_vecs:m],
                                      3)
        print("the classifier came back with: %d, the real answer is : %d", classifier_result, dating_data_labels[i])

        if classifier_result != dating_data_labels[i]:
            error_count += 1.0
    print("the total error rate is : %f", (error_count / float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm, dating_labels, 3)
    print("you will probably like this person: %s", result_list[classifier_result - 1])


# 数字识别
def img2vector(file_name):
    return_vect = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])

    return return_vect


def handwriting_class_test():
    training_set_path = '/Users/wangxiao15/Desktop/machinelearninginaction/Ch02/digits/trainingDigits'
    test_set_path = '/Users/wangxiao15/Desktop/machinelearninginaction/Ch02/digits/testDigits'
    hw_labels = []
    training_file_list = listdir(training_set_path)
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector(training_set_path + '/' + file_name_str)
    test_file_list = listdir(test_set_path)
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector(test_set_path + '/' + file_name_str)
        classifier_result = classify0(vector_under_test,
                                      training_mat,
                                      hw_labels,
                                      3)
        print("the classifier came back with : %d, the real answer is : %d", classifier_result, class_num_str)

        if classifier_result != class_num_str:
            error_count += 1
    print("\nthe totalnube of errors is : %d", error_count)
    print("\nthe total error rate is : %f", error_count / float(m_test))


if __name__ == '__main__':
    handwriting_class_test()

