from __future__ import print_function

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from os import path
from gcn.smooth import smooth
import os
import networkx as nx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
import tensorflow.compat.v1 as tf1


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def split_dataset(onehot_labels, labelids, train_size, test_size, validation_size, validate=True, shuffle=True):
    idx = np.arange(len(onehot_labels))
    idx_left = []
    # idx_val = []
    if shuffle:
        np.random.shuffle(idx)
    if isinstance(train_size, int):
        assert train_size > 0, "train size must bigger than 0."
        no_class = onehot_labels.shape[1]  # number of class
        train_size = [train_size for i in range(no_class)]
        idx_train = []
        count = [0 for i in range(no_class)]
        label_each_class = train_size
        next = 0
        for i in idx:
            if count == label_each_class:
                break
            next += 1
            j = labelids[i]
            if count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1
            else:
                idx_left.append(i)

        idx = np.concatenate([idx_left, idx[next:]])
        next = 0
        # idx_test = np.array(idx_test, dtype=idx.dtype)
        if validate:
            assert isinstance(validation_size, int)
            if validation_size:
                assert next + validation_size < len(idx), "Too many train data, no data left for validation."
                idx_val = idx[next:next + validation_size]
                next = next + validation_size
            else:
                idx_val = idx[next:]

            assert next < len(idx), "Too many train and validation data, no data left for testing."
            if test_size:
                assert next + test_size < len(idx)
                idx_test = idx[-test_size:]
            else:
                idx_test = idx[next:]
        else:
            assert next < len(idx), "Too many train data, no data left for testing."
            if test_size:
                assert next + test_size < len(idx)
                idx_test = idx[-test_size:]
            else:
                idx_test = np.concatenate([idx_test, idx[next:]])
            idx_val = idx_test
    else:
        # train
        assert isinstance(train_size, float)
        assert 0 < train_size < 1, "float train size must be between 0-1"
        labels_of_class = [0]
        train_size = int(len(idx) * train_size)
        next = 0
        try_time = 0
        while np.prod(labels_of_class) == 0 and try_time < 100:
            np.random.shuffle(idx)
            idx_train = idx[next:next + train_size]
            labels_of_class = np.sum(onehot_labels[idx_train], axis=0)
            try_time = try_time + 1
        next = train_size

        # validate
        if validate:
            assert isinstance(validation_size, float)
            validation_size = int(len(idx) * validation_size)
            idx_val = idx[next: next + validation_size]
            next += validation_size
        else:
            idx_val = idx[next:]

        # test
        if test_size:
            assert isinstance(test_size, float)
            test_size = int(len(idx) * test_size)
            idx_test = idx[next: next + test_size]
        else:
            idx_test = idx[next:]
    idx_train = np.array(idx_train)
    return idx_train, idx_val, idx_test


def load_data(dataset_str, train_size, validation_size, model_config, shuffle=True):
    if dataset_str in ['ogbn-arxiv']:
        from ogb.nodeproppred import NodePropPredDataset
        dataset = NodePropPredDataset('ogbn-arxiv', root='data/')
        l = dataset.labels.flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1

        n = dataset.graph['num_nodes']
        edge_index = dataset.graph['edge_index']
        adj = sp.csc_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n))
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        features = dataset.graph['node_feat']
        split = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split['train'], split['valid'], split['test']
        data = sio.loadmat('data/ogbn_arxiv/{}.mat'.format(dataset_str, dataset_str))
        features = data['X']
        # idx_train, idx_val, idx_test = split_dataset(labels, l, train_size, model_config['test_size'], validation_size,
        #                                          validate=model_config['validate'], shuffle=shuffle)
    elif dataset_str in ['large_cora', '20news', 'wiki'] or dataset_str.startswith('webkb'):
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        adj = data['G'] if 'G' in data else sp.eye(features.shape[0], dtype=features.dtype)
        # edge_statistic = adj.dot(labels).T.dot(labels)
        # G = nx.from_scipy_sparse_matrix(adj)
        # com = [*nx.connected_components(G)]
        idx_train, idx_val, idx_test = split_dataset(labels, l, train_size, model_config['test_size'], validation_size,
                                                 validate=model_config['validate'], shuffle=shuffle)
    elif dataset_str == 'Hindroid':
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['A']
        P = data['P']
        data['P'] = P + P + 10*sp.eye(P.shape[0])
        adj = sp.eye(features.shape[0])
        idx_train, idx_val, idx_test = split_dataset(labels, l, train_size, model_config['test_size'], validation_size,
                                                 validate=model_config['validate'], shuffle=shuffle)
    elif dataset_str == 'malware':
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        if model_config['F']:
            # features = data['X']
            features = data['X2']
            features = sp.hstack([data['X'], data['X2']])
            model_config['F'] = None
        else:
            features = data['X']
        adj = data['G'] if 'G' in data else sp.eye(features.shape[0], dtype=features.dtype)
        # edge_statistic = adj.dot(labels).T.dot(labels)
        # G = nx.from_scipy_sparse_matrix(adj)
        # com = [*nx.connected_components(G)]
        idx_train, idx_val, idx_test = split_dataset(labels, l, train_size, model_config['test_size'], validation_size,
                                                 validate=model_config['validate'], shuffle=shuffle)

    elif dataset_str in ['malware2']:
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(data['labels']) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        Perm = data['Perm']
        # features = data['Perm']
        # adj = Perm@Perm.T
        adj = construct_knn_graph(Perm, 3)
        # adj = data['G']
        B = data['coblock']
        B = normalize(B.T, norm='l1').T
        # F = data['coblock']

        if 'right_repeat' in model_config:
            k = model_config['right_repeat']
            if k: features = (1 - k) * features + k * features.dot(B)
        idx_train, idx_val, idx_test = split_dataset(labels, l, train_size, model_config['test_size'], validation_size,
                                                 validate=model_config['validate'], shuffle=shuffle)
    elif dataset_str == 'malware2':
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        Perm = data['Perm']
        # adj = Perm@Perm.T
        adj = construct_knn_graph(Perm, 3)
        idx_train, idx_val, idx_test = split_dataset(labels, l, train_size, model_config['test_size'], validation_size,
                                                 validate=model_config['validate'], shuffle=shuffle)
    else:
        raise ValueError('Unknown dataset {}'.format(dataset_str))

    if model_config['F']:
        F_name = model_config['F']
        F_name, modifier = (F_name.split('_') + [''])[:2]
        if F_name == 'CE':
            F = features.T.dot(labels) / labels.sum(0) ** 0.5
            F = construct_knn_graph(F, 10)
        else:
            F = data[F_name]
        F.eliminate_zeros()
        if modifier == 'row':
            F = normalize(F, 'l1', 1)
        elif modifier == 'col':
            F = normalize(F, 'l1', 0)
        elif modifier == 'sym':
            F = symmetric_normalize(F)
        elif modifier == 'dbs':
            fetch = np.ones(features.T.shape[0], dtype=np.bool)
            new_features = np.zeros(features.T.shape, dtype=features.dtype)
            new_features[fetch], smoothing_time = smooth(features.T, F.T, 'ap_appro',
                                                         {'cache': False, 'smooth_alpha': 1.,},fetch=fetch)
            features = new_features.T
            F = None
        elif modifier == '':
            pass
        else:
            raise ValueError(modifier)
        # if modifier != 'dbs':
        #     F = (F + sp.eye(*F.shape)) / 2
    else:
        F = None

    print('labels of each class : ', np.sum(labels[idx_train], axis=0))
    # idx_val = idx[len(idx) * train_size // 100:len(idx) * (train_size // 2 + 50) // 100]
    # idx_test = idx[len(idx) * (train_size // 2 + 50) // 100:len(idx)]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, F


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return tf1.SparseTensorValue(coords, values, np.array(shape, dtype=np.int64))

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def symmetric_normalize(a):
    row = np.array(a.sum(1)).flatten()
    row = np.power(row, -0.5)
    row[np.isinf(row)] = 0.

    col = np.array(a.sum(0)).flatten()
    col = np.power(col, -0.5)
    col[np.isinf(col)] = 0.

    if sp.issparse(a):
        row = sp.diags(row)
        col = sp.diags(col)
        return row.dot(a).dot(col)
    else:
        a = np.array(a)
        n,m = a.shape
        row = row.reshape(n, 1)
        col = col.reshape(1, m)
        return a*row*col


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
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
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
    adj = normalize_adj(adj, type=type)
    return sparse_to_tuple(adj)


def absorption_probability(W, alpha, stored_A=None, column=None):
    store_dir = 'cache/'
    stored_A = store_dir + stored_A
    try:
        # raise Exception('DEBUG')
        A = np.load(stored_A + str(alpha) + '.npz')['arr_0']
        print('load A from ' + stored_A + str(alpha) + '.npz')
        if column is not None:
            P = np.zeros(W.shape)
            P[:, column] = A[:, column]
            return P
        else:
            return A
    except:
        # W=sp.csr_matrix([[0,1],[1,0]])
        # alpha = 1
        print('Calculate absorption probability...')
        W = W.copy().astype(np.float32)
        D = W.sum(1).flat
        L = sp.diags(D, dtype=np.float32) - W
        # L = L.dot(L)
        L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
        L = sp.csc_matrix(L)
        # print(np.linalg.det(L))

        if column is not None:
            A = np.zeros(W.shape)
            # start = time.time()
            A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
            # print(time.time()-start)
            return A
        else:
            # start = time.time()
            A = slinalg.inv(L).toarray()
            # print(time.time()-start)
            if stored_A:
                np.savez(stored_A + str(alpha) + '.npz', A)
            return A


def LP(adj, alpha, y_train, y_test):
    from gcn.smooth import smooth
    prediction, time = smooth(y_train, adj, 'ap_appro', {
        'cache': False,
        'smooth_alpha': 1 / alpha,
    }, stored_A=None, fetch=None)
    predicted_labels = np.argmax(prediction, axis=1)
    prediction = np.zeros(prediction.shape)
    prediction[np.arange(prediction.shape[0]), predicted_labels] = 1

    test_acc = np.sum(prediction * y_test) / np.sum(y_test)
    test_acc_of_class = np.sum(prediction * y_test, axis=0) / np.sum(y_test, axis=0)
    # print(test_acc, test_acc_of_class)
    return test_acc, test_acc_of_class, prediction


def construct_knn_graph(features, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    adj = nbrs.kneighbors_graph()
    adj = adj + adj.T
    adj[adj != 0] = 1
    return adj


def construct_feed_dict(features, support, labels, labels_mask, dropout, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape if isinstance(features,
                                                                                            tf1.SparseTensorValue) else [
        0]})
    feed_dict.update({placeholders['dropout']: dropout})
    return feed_dict


def preprocess_model_config(model_config):
    if model_config['Model'] != 'LP':
        model_config['connection'] = list(model_config['connection'])
        # judge if parameters are legal
        for c in model_config['connection']:
            if c not in ['c', 'd', 'r', 'f', 'C']:
                raise ValueError(
                    'connection string specified by --connection can only contain "c", "d", "r", "f", "C" but "{}" found'.format(
                        c))
        for i in model_config['layer_size']:
            if not isinstance(i, int):
                raise ValueError('layer_size should be a list of int, but found {}'.format(model_config['layer_size']))
            if i <= 0:
                raise ValueError('layer_size must be greater than 0, but found {}' % i)
        if not len(model_config['connection']) == len(model_config['layer_size']) + 1:
            raise ValueError('length of connection string should be equal to length of layer_size list plus 1')

    # Generate name
    if not model_config['name']:
        model_name = model_config['Model']

        if model_config['validate']:
            model_name += '_validate'

        if model_config['G']:
            model_name += '_G'
            if type(model_config['G']) == int:
                model_name += str(model_config['G'])

        if model_config['F']:
            model_name += '_' + str(model_config['F'])

        if model_config['Model'] in 'LP':
            model_name += '_alpha_' + str(model_config['alpha'])

        if model_config['dataset'] in ['Hindroid', 'malware2']:
            if 'right_repeat' in model_config:
                model_name += '_k' + str(model_config['right_repeat'])

        model_config['name'] = model_name

    # Generate logdir
    if model_config['logging'] and not model_config['logdir']:
        train_size = '{}_train'.format(model_config['train_size'])
        i = 0
        while True:
            logdir = path.join('log', model_config['dataset'],
                               train_size, model_config['name'], 'run' + str(i))
            i += 1
            if not path.exists(logdir):
                break
        model_config['logdir'] = logdir
