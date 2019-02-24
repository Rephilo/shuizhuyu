import numpy as np
from sklearn.naive_bayes import GaussianNB


def bayes_test():
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = GaussianNB(priors=None)
    clf.fit(x, y)
    print(clf.predict([[-0.8, -1]]))


if __name__ == '__main__':
    bayes_test()
