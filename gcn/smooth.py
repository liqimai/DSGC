from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import hashlib
import warnings
import time
# from gcn.utils import normalize_adj

cupy_is_available = False
try:
    import cupy as cp
    cupy_is_available = True
except ImportError:
    warnings.warn("cupy is not imported, some operations may not use GPU.", ImportWarning)

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

def md5(*args):
    fingerprint = b''
    for arg in args:
        if isinstance(arg, str):
            fingerprint += arg.encode()
        elif isinstance(arg, (float, int, complex, list, tuple, dict, bool)) or arg is None:
            fingerprint += str(arg).encode()
        elif isinstance(arg, np.ndarray):
            fingerprint += arg.tobytes()
        else:
            raise NotImplementedError('Type:', str(type(arg)), 'Value:', arg)
    return hashlib.md5(fingerprint).hexdigest()

def smooth(features, adj, smoothing, model_config, stored_A=None, fetch=None):
    def cache_file(*args):
        return "data/cache/{}.npz".format(md5(*args))

    print(smoothing, 'Smoothing...')
    cache = True if model_config['cache'] else None
    if smoothing is None:
        return features, 0.
    elif smoothing == 'taubin':
        if cache:
            cache = cache_file(model_config['dataset'], model_config['taubin_lambda'], model_config['taubin_mu'],
                                model_config['taubin_repeat'], fetch)
        return taubin_smoothing(adj, model_config['taubin_lambda'], model_config['taubin_mu'],
                                model_config['taubin_repeat'], features, fetch=fetch, cache=cache)
    elif smoothing == 'ap_appro':
        k = int(np.ceil(4 / model_config['smooth_alpha']))
        if cache:
            cache = cache_file(model_config['dataset'], model_config['smooth_alpha'], k, fetch)
        return ap_approximate(adj, features, model_config['smooth_alpha'], k, fetch=fetch, cache=cache)
    else:
        raise ValueError("smoothing must be one of 'poly' | 'ap' | 'taubin' | 'test21' | 'test27' ")

import os
def npz_cache(func):
    def func_with_cache(*args, cache=None, **kwargs):
        if cache is not None and os.path.isfile(cache):
            print('loading', cache, '...')
            loader = np.load(cache)
            if loader['multi_return']:
                n = len(list(loader.iterkeys()))-1
                results = (loader['arr_'+str(i)] for i in range(n))
            else:
                results = loader['results']
            print('success', cache)
        else:
            results = func(*args, **kwargs)
            if cache is not None:
                print('save to', cache)
                if isinstance(results, (list, tuple)):
                    np.savez_compressed(cache, *results, multi_return=True)
                else:
                    np.savez_compressed(cache, results=results, multi_return=False)
        return results
    return func_with_cache

def gpu_taubin_smoothing(step_transformor, features, repeat, fetch):
    #TODO: transfer sparse features to GPU
    #TODO: only fetch necessary data
    smooth_time = 0
    step_transformor = cp.sparse.csr_matrix(step_transformor)
    step_transformor.sum_duplicates()
    tile_width = 1024**3//4//4//features.shape[0]
    #initialzie new_features
    if fetch is None:
        new_features = features
    else:
        new_features = features[fetch]
    if sp.issparse(new_features):
        new_features = new_features.todense()

    for i in range(0, features.shape[1], tile_width):
        low = i
        high = min(features.shape[1], i+tile_width)
        # transfer data to GPU
        if sp.issparse(features):
            tile = cp.sparse.csr_matrix(features[:, low:high])
            tile = tile.todense()
        else:
            tile = cp.asarray(features[:, low:high])
        tile = cp.asfortranarray(tile)
        tile.device.synchronize()

        # calculate
        begin = time.time()
        for i in range(repeat):
            tile = cp.cusparse.csrmm2(step_transformor, tile, tile)
            # tile = step_transformor.dot(tile)
        tile.device.synchronize()
        smooth_time += time.time()-begin

        # fetch
        if fetch is None:
            new_features[:, low:high] = tile.get()
        else:
            new_features[:, low:high] = tile[fetch].get()
    return new_features, smooth_time

@npz_cache
def taubin_smoothing(adj, lam, mu, repeat, features, fetch=None):
    smooth_time = 0
    n = adj.shape[0]
    # adj = normalize(adj + sp.eye(adj.shape[0]), norm='l1', axis=1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]), 'rw')
    smoothor = sp.eye(n) * (1 - lam) + lam * adj
    inflator = sp.eye(n) * (1 - mu) + mu * adj
    step_transformor = smoothor * inflator
    step_transformor = step_transformor.astype(np.float32)
    features = features.astype(np.float32)
    if cupy_is_available:
        print('USE GPU')
        features, smooth_time = gpu_taubin_smoothing(step_transformor, features, repeat, fetch)
    else:
        if sp.issparse(features):
            features = features.toarray()
        begin = time.time()
        for i in range(repeat):
            features = step_transformor.dot(features)
        smooth_time += time.time()-begin
        if fetch is not None:
            features = features[fetch]
    if sp.issparse(features):
        features = features.toarray()
    return features, smooth_time

# RNM
def taubin_smoothor(adj, lam, mu, repeat):
    n = adj.shape[0]
    # adj = normalize(adj + sp.eye(adj.shape[0]), norm='l1', axis=1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]), 'rw')
    smoothor = sp.eye(n) * (1 - lam) + lam * adj
    inflator = sp.eye(n) * (1 - mu) + mu * adj
    step_transformor = smoothor * inflator
    transformor = sp.eye(n)
    base = step_transformor
    while repeat != 0:
        if repeat % 2:
            transformor *= base
        base *= base
        repeat //= 2
        # print(repeat)
    return transformor

def gpu_ap_approximate(adj, features, alpha, k, fetch):
    features = features.astype(np.float32)
    if fetch is None:
        new_features = features
    else:
        new_features = features[fetch]
    if sp.issparse(new_features):
        new_features = new_features.todense()

    smooth_time = 0
    adj = cp.sparse.csr_matrix(adj)
    adj.sum_duplicates()
    tile_width = 1024**3//4//2//features.shape[0]
    for i in range(0, features.shape[1], tile_width):
        low = i
        high = min(features.shape[1], i+tile_width)
        # transfer data to GPU
        if sp.issparse(features):
            new_features_tile = cp.sparse.csr_matrix(features[:, low:high])
            features_tile = cp.sparse.csr_matrix(features[:, low:high])
            new_features_tile = new_features_tile.todense()
            features_tile = features_tile.todense()
        else:
            new_features_tile = cp.asarray(features[:, low:high])
            features_tile = cp.asarray(features[:, low:high])
        new_features_tile = cp.asfortranarray(new_features_tile)
        new_features_tile.device.synchronize()

        #calculate
        begin = time.time()
        for _ in range(k - 1):
            # new_feature = adj.dot(new_feature) + features
            new_features_tile = cp.cusparse.csrmm2(adj, new_features_tile, new_features_tile)
            new_features_tile += features_tile
        new_features_tile *= alpha / (alpha + 1)
        new_features_tile.device.synchronize()
        smooth_time += time.time()-begin

        # fetch
        if fetch is None:
            new_features[:, low:high] = new_features_tile.get()
        else:
            new_features[:, low:high] = new_features_tile[fetch].get()
    return new_features, smooth_time

@npz_cache
def ap_approximate(adj, features, alpha, k, fetch=None):
    smooth_time = 0
    # adj = normalize(adj + sp.eye(adj.shape[0]), 'l1', axis=1) / (alpha + 1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]), 'sym') / (alpha + 1)
    adj = adj.astype(np.float32)
    if cupy_is_available:
        print('USE GPU')
        new_feature, smooth_time = gpu_ap_approximate(adj, features, alpha, k, fetch=fetch)
    else:
        if sp.issparse(features):
            features = features.toarray()
        features = features.astype(np.float32)
        new_feature = np.zeros(features.shape, dtype=features.dtype)
        begin = time.time()
        for _ in range(k):
            new_feature = adj.dot(new_feature)
            new_feature += features
        new_feature *= alpha / (alpha + 1)
        smooth_time += time.time()-begin
        if fetch is not None:
            new_feature = new_feature[fetch]
    return new_feature, smooth_time

# @npz_cache
# def md(adj, features, alpha, k, repeat):
#     smoothing_time = time.time()
#     for i in range(repeat):
#         features, _ = ap_approximate(adj, features, alpha, int(np.ceil(4 / alpha)))
#         adj = construct_knn_graph(features, k)
#     return adj, time.time()-smoothing_time


# def test21(adj, alpha, beta, stored_a=None):
#     p = absorption_probability(adj + sp.eye(adj.shape[0]), alpha, stored_A=stored_a)
#     p *= alpha
#     # p = (p > (beta / alpha)).astype(np.float32)
#     lines = np.min([100, p.shape[0]])
#     idx = np.arange(p.shape[0])
#     np.random.shuffle(idx)
#     idx = idx[:lines]
#     p_flat = p[idx].flat
#     p_index = np.argsort(p_flat)
#     p_acc = np.add.accumulate(p_flat[p_index]) / lines
#     percentage = 1 - p_acc[-beta * lines]
#     # num = np.sum(p_acc <= (1-beta))
#     # gate = p_flat[p_index[np.maximum(num-1, 0)]]
#
#     num = beta * lines
#     gate = p_flat[p_index[len(p_index) - num]]
#     p = (p > [gate]).astype(np.float32)
#
#     # global all_labels
#     c = np.argmax(all_labels, axis=1)
#     c = c == np.expand_dims(c, 1)
#     num = np.sum(p)
#     print("neighbor accuracy = ", np.sum(c * p) / num, 'average #neighbors = ', num / p.shape[0], 'energy reserved=',
#           percentage)
#     # normalize(p, norm='l1', axis=1, copy=False)
#     return sp.csr_matrix(p / beta)


# def test27(adj, features, alpha, beta, stored_a=None):
#     p = absorption_probability(adj + sp.eye(adj.shape[0]), alpha, stored_A=stored_a)
#     # np.sort(p)
#     p *= alpha
#     # p = np.array([
#     #     [0.1, 0.5, 0.4],
#     #     [0.2, 0.7, 0.1],
#     #     [0.4, 0.3, 0.3]
#     # ])
#     p_index = np.argsort(p, axis=1)
#     p_acc = np.add.accumulate(np.sort(p, axis=1), axis=1)
#     # plt.plot(np.add.accumulate(np.sort(p, axis=None)))
#     # plt.grid()
#     # plt.show()
#     num = np.sum(p_acc <= (1 - beta), axis=1)
#     gate = p[np.arange(p.shape[0]), p_index[np.arange(p.shape[0]), np.maximum(num - 1, 0)]]
#     # p[p <= [gate]]=0
#     p = (p > [gate]).astype(np.float32)
#     p = normalize(p, norm='l1', axis=1, copy=False)
#     return sp.csr_matrix(p * features)

def model_name_modify(model_name, model_config):
    if model_config['smoothing'] == 'ap':
        model_name += '_' + 'ap_smoothing' + '_' + str(model_config['smooth_alpha'])
    if model_config['smoothing'] == 'ap_appro':
        model_name += '_' + 'ap_appro' + '_' + str(model_config['smooth_alpha'])
    elif model_config['smoothing'] == 'test21':
        model_name += '_' + 'test21' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['beta'])
    elif model_config['smoothing'] == 'test21_norm':
        model_name += '_' + 'test21_norm' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['beta'])
    elif model_config['smoothing'] == 'test27':
        model_name += '_' + 'test27' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['beta'])
    elif model_config['smoothing'] == 'poly':
        model_name += '_' + 'poly_smoothing'
        for a in model_config['poly_parameters']:
            model_name += '_' + str(a)
    elif model_config['smoothing'] == 'taubin':
        model_name += '_taubin' + str(model_config['taubin_lambda']) \
                      + '_' + str(model_config['taubin_mu']) \
                      + '_' + str(model_config['taubin_repeat'])
    elif model_config['smoothing'] == 'sparse_encoding':
        model_name += '_sparse_encoding'
    elif model_config['smoothing'] is 'manifold_denoising':
        model_name += '_manifold_denoising' + '_' + str(model_config['smooth_alpha']) + '_' + str(
            model_config['md_repeat'])
    elif model_config['smoothing'] is None:
        pass
    else:
        raise ValueError('invalid smoothing')

    return model_name