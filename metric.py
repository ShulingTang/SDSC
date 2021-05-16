import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn import cluster
from scipy.sparse.linalg import svds, eigs
from cvxopt import matrix, solvers
from sklearn.preprocessing import normalize
import torch
import matlab.engine

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score

def get_Coefficient(x, kmeansNum):
    """
    In this part, first conduct kmeans to obtain M anchor points,
    then construct a coefficient matrix C through the anchor points
    and all the data.
    :param x: sampled data
    :param kmeansNum:anchor graph
    :return C:Coefficient Matrix
    """
    # x.shape = (n, d), m.shape = (m, d)
    alpha = 1.0
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    # x = x.numpy()
    m = KMeans(n_clusters=kmeansNum, random_state=0).fit(x)
    num, dim = x.shape
    m = m.cluster_centers_   # m.shape = (m, d)

    # 该部分为翻译的matlab代码
    h = 2*alpha*np.identity(kmeansNum)+2*np.matmul(m, m.T)
    h = matrix(1/2*(h+h.T))  # h.shape = (m, m)
    bb = x.T   # B.shape = (d, n)
    z = []
    l = matrix(np.ones(kmeansNum))

    # I = matrix(I.T)
    o = matrix(0.0)
    for i in range(num):
        # fi.shape = (m ,1)
        fi = np.matmul(-2*(bb[:, i]).T, m.T)
        # fi.type = ndarray　shape = {tuple}(m,)
        fi = fi.astype(np.float64)
        fi = matrix(fi.reshape(1, kmeansNum))
        fi = matrix(fi)
        zi = solvers.qp(h, fi.T, G=None, h=None, A=l.T, b=o, kktsolver=None)
        # print(zi.shape)
        zi = np.array(zi['x']).reshape(kmeansNum)
        z = np.r_[z, zi]
    z = z.reshape(num, kmeansNum)  # z.shape = (n, m)
    z = z.astype(np.float32)
    z = torch.from_numpy(z).to('cuda')
    m = torch.from_numpy(m).to('cuda')
    return z, m

class get_m(nn.Module):
    def __init__(self, n, kmeansNum):
        super(get_m, self).__init__()
        # torch.nn.Paramater 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.Chat = nn.Parameter(1.0 * torch.ones(kmeansNum, n, dtype=torch.float32), requires_grad=True)
    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Chat, x)
        return y

class SelfExpression(nn.Module):
    def __init__(self, n, kmeansNum):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0 * torch.ones(n, kmeansNum, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        Coefficient = F.softmax(self.Coefficient, 1)
        y = torch.matmul(Coefficient, x)
        return y

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)

def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp

def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    # U.shape=(n,r)
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L
'''

def spectral_clustering_quick(C, Chat, K, d=None, alpha=None, ro=None):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    n, m = C.shape
    # C = np.matmul(C, C.T)
    C = C.tolist()

    # y = eng.spectral(C, K, n, m)
    # y = np.array(y)
    # y = torch.from_numpy(y).to('cuda')

    U = eng.mySVD(C, K, n, m)
    # U, sig, V = result
    # y = eng.litekmeans(U, K)
    # labels = litekmeans(U, k, 'MaxIter', 100, 'Replicates', 10)
    eng.quit()
    y = KMeans(n_clusters=K, random_state=0).fit(U)
    y = y.labels_
    # print(y)
    return y
def spectral_clustering_pre(C,  K, d, alpha, ro):
    # C = torch.from_numpy(C).to('cuda')
    # Chat = torch.from_numpy(Chat).to('cuda')
    C = np.matmul(C, C.T)
    # C = C.detach().cpu().numpy()
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y
'''
def spectral_clustering(SC_method, C, Chat, K, d=None, alpha=None, ro=None):
    if SC_method:
        eng = matlab.engine.start_matlab()
        n, m = C.shape
        # C = np.matmul(C, C.T)
        C = C.tolist()
        U = eng.mySVD(C, K, n, m)
        # U, sig, V = result
        # y = eng.litekmeans(U, K)
        # labels = litekmeans(U, k, 'MaxIter', 100, 'Replicates', 10)
        eng.quit()
        y = KMeans(n_clusters=K, random_state=0).fit(U)
        y = y.labels_
        # print(y)
        return y
    else:
        C = np.matmul(C, C.T)
        # C = C.detach().cpu().numpy()
        C = thrC(C, alpha)
        y, _ = post_proC(C, K, d, ro)
        return y
