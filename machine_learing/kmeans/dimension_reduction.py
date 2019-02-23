import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState


def pca_test():
    data = load_iris()
    y = data.target
    x = data.data
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(x)

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []

    for i in range(len(reduced_X)):
        if y[i] == 0:
            red_x.append(reduced_X[i][0])
            red_y.append(reduced_X[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_X[i][0])
            blue_y.append(reduced_X[i][1])
        else:
            green_x.append(reduced_X[i][0])
            green_y.append(reduced_X[i][1])

    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()


if __name__ == '__main__':
    n_row, n_col = 2, 3
    n_components = n_row * n_col
    image_shape = (64, 64)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
    faces = dataset.data


    def plot_gallery(title, images, n_col=n_col, n_row=n_row):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)

        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())

            plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                       interpolation='nearest',
                       vimn=vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


    plot_gallery("First centered Olivetti faces", faces[:n_components])

    estimators = [
        ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=6, whiten=True)),

        ('Non-negative components - NMF',
         decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))
    ]
    for name, estimator in estimators:
        print("Extracting the top %d %s..." % (n_components, name))
        print(faces.shape)
        estimator.fit(faces)
        components_ = estimator.components_
        plot_gallery(name, components_[:n_components])

    plt.show()




