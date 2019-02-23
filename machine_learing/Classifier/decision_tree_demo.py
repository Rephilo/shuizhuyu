from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def decision_tree_test():
    clf = DecisionTreeClassifier()
    iris = load_iris()
    x = cross_val_score(clf, iris.data, iris.target, cv=10)
    print(x)


if __name__ == '__main__':
    decision_tree_test()
