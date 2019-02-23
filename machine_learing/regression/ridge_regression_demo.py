import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation


def ridge_regression_test():
    data = np.genfromtxt('data.txt')
    plt.plot(data[:, 4])
    x = data[:, :4]
    y = data[:, 4]
    poly = PolynomialFeatures(6)
    x = poly.fit_transform(x)

    # train_set_x,test_set_x,train_set_y,test_set_y=cross_validation.train_test.split
