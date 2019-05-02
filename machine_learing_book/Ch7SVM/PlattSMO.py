from numpy import *


class optStruct:
    def __init__(self, data_mat_in, class_labels, C, toler):
        self.X = data_mat_in
        self.label_mat = class_labels
        self.C = C
        self.tol = toler
        self.m = shape(data_mat_in)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.ecache = mat(zeros((self.m, 2)))


def calc_ek(os, k):
    fXk = float(multiply(os.alphas, os.label_mat)).T(os.X * os.X[k, :].T) + os.b
    ek = fXk - float(os.label_mat[k])
    return ek


def selectJ(i, os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcacheList = nonzero(os.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calc_ek(os, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calc_ek(os, j)
    return j, Ej


def updateEk(os, k):
    Ek = calc_ek()
    os.eCache[k] = [1, Ek]
