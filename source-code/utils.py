import math
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse.linalg import svds
from scipy.spatial import distance
from sklearn import cluster
from sklearn.preprocessing import normalize as nor

eps = 2.2204e-16


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))  # 奇异值分解，r为奇异值数量；返回的是左奇异值向量，奇异值（对角线的值）
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)  # 返回的矩阵为对角线值为S的元素，其余位置为0
    U = U.dot(S)  # 计算U*S
    U = nor(U, norm='l2', axis=1)  # l2范式处理矩阵
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    # K为聚类个数，初始化中心数在默认值10，affinity指定相似度矩阵的计算方式
    spectral.fit(L)  # 返回聚类对象本身
    grp = spectral.fit_predict(L) + 1  # 返回数据集预测标签
    return grp, L


def Conformal_mapping(X):
    a = np.zeros(X.shape[1])
    for i in range(0, X.shape[1]):
        a[i] = np.dot(X.T[i], np.ones(X.shape[0])) / X.shape[0]
    a = np.matrix(a)
    dists = distance.cdist(X, a, 'euclidean')
    R = np.max(dists)
    Xf = np.zeros_like(X)
    Xe = np.zeros(len(X))
    for i in range(0, len(X)):
        Xe[i] = R * ((np.dot(X[i] ** 2, np.ones(X.shape[1])) - math.pow(R, 2)) / (
                (np.dot(X[i] ** 2, np.ones(X.shape[1]))) + math.pow(R, 2)))
        Xf[i] = R * (2 * R / ((np.dot(X[i] ** 2, np.ones(X.shape[1]))) + math.pow(R, 2)) * X[i])

    X1 = np.column_stack((Xf, Xe))
    return X1


def spectralclustering(data, n_clusters):
    N = data.shape[0]
    maxiter = 1000  # max iteration times
    replic = 100  # number of time kmeans will be run with diff centroids

    DN = np.diag(1 / np.sqrt(np.sum(data, axis=0) + eps))
    lapN = np.eye(N) - DN.dot(data).dot(DN)
    U, A, V = np.linalg.svd(lapN)
    V = V.T
    kerN = V[:, N - n_clusters:N]
    normN = np.sum(kerN ** 2, 1) ** 0.5
    kerNS = (kerN.T / (normN + eps)).T
    # kmeans
    clf = KMeans(n_clusters=n_clusters, max_iter=maxiter, n_init=replic)
    return clf.fit_predict(kerNS)
