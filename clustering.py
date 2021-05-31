import scipy.io as sio
import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        #adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label




if __name__ == '__main__':

    dataset = 'webkb_washington'
    data = sio.loadmat('data/{}.mat'.format(dataset))
    feature = data['X']
    if sp.issparse(feature):
        feature = feature.todense()

    adj = data['G']
    # Using PMI
    F = data['PMI']
    F = preprocess_adj(F)
    # Using Emb
    # F = data['Emb']
    # F = preprocess_adj(F)
    F = (sp.eye(F.shape[0]) + F)/2
    if sp.issparse(F):
        F = F.todense()
    gnd = data['labels']
    gnd = gnd[0, :]
    k = len(np.unique(gnd))
    adj = sp.coo_matrix(adj)
    rep = 10


    acc_list = []
    nmi_list = []
    f1_list = []
    stdacc_list = []
    stdnmi_list = []
    stdf1_list = []

    adj_normalized = preprocess_adj(adj, type='rw')

    ac = np.zeros(rep)
    nm = np.zeros(rep)
    f1 = np.zeros(rep)

    feature = adj_normalized.dot(feature)
    feature = adj_normalized.dot(feature)
    feature = feature.dot(F)
    u, s, v = sp.linalg.svds(feature, k=k, which='LM')




    for i in range(rep):
        kmeans = KMeans(n_clusters=k).fit(u)
        predict_labels = kmeans.predict(u)
            
        cm = clustering_metrics(gnd, predict_labels)
        ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()

        
    acc_means = np.mean(ac)
    acc_stds = np.std(ac)
    nmi_means = np.mean(nm)
    nmi_stds = np.std(nm)
    f1_means = np.mean(f1)
    f1_stds = np.std(f1)

        
    print('acc_mean: {}'.format(acc_means),
          'acc_std: {}'.format(acc_stds),
          'nmi_mean: {}'.format(nmi_means),
          'nmi_std: {}'.format(nmi_stds),
          'f1_mean: {}'.format(f1_means),
          'f1_std: {}'.format(f1_stds))

        
        


