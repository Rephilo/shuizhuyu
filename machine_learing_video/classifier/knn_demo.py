from sklearn.neighbors import KNeighborsClassifier


def knn_test():
    x = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    print(neigh.predict([[1.2]]))


if __name__ == '__main__':
    knn_test()
