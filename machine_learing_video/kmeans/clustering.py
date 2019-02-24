import numpy as np
from sklearn.cluster import KMeans,DBSCAN


def analyze_city():
    data, city_name = load_data('city.txt')
    km = KMeans(n_clusters=3)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_, axis=1)
    print(expenses)
    city_cluster = [[], [], []]
    for i in range(len(city_name)):
        city_cluster[label[i]].append(city_name[i])
    for i in range(len(city_cluster)):
        print("Expenses:%.2f" % expenses[i])
        print(city_cluster[i])


def load_data(file_path):
    fr = open(file_path, 'r+')
    lines = fr.readlines()
    ret_data = []
    ret_city_name = []
    for line in lines:
        items = line.strip().split(",")
        ret_city_name.append(items[0])
        ret_data.append([float(items[i])
                         for i in range(1, len(items))])
        for i in range(1, len(items)):
            return ret_data, ret_city_name


