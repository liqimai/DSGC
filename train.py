from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
from scipy import sparse
from gcn.utils import (
    construct_feed_dict, preprocess_adj,
    load_data, sparse_to_tuple, LP, symmetric_normalize)
from gcn.smooth import smooth
from gcn.models import GCN_MLP
from config import configuration, args
from cvr import cvr, inter_intra_variance, QM
from tabulate import tabulate

def train(model_config, sess):
    # Print model_config
    very_begining = time.time()
    print('',
          'name           : {}'.format(model_config['name']),
          'logdir         : {}'.format(model_config['logdir']),
          'dataset        : {}'.format(model_config['dataset']),
          'train_size     : {}'.format(model_config['train_size']),
          'learning_rate  : {}'.format(model_config['learning_rate']),
          'logging        : {}'.format(model_config['logging']),
          sep='\n')

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, F = \
        load_data(model_config['dataset'],train_size=model_config['train_size'],
                  validation_size=model_config['validation_size'],
                  model_config=model_config, shuffle=model_config['shuffle'])

    def sample_graph_convolution(features, k):
        # fetch = train_mask+val_mask+test_mask
        # new_features = np.zeros(features.shape, dtype=features.dtype)
        new_features, smoothing_time = smooth(features, adj, 'taubin',
                                                     {
                                                        'taubin_lambda': 1,
                                                        'taubin_mu': 0,
                                                        'taubin_repeat': k,
                                                        'cache'       : False,
                                                     })
        return new_features, smoothing_time

    def sample_graph_convolution_lp(features, alpha):
        fetch = train_mask+val_mask+test_mask
        new_features = np.zeros(features.shape, dtype=features.dtype)
        new_features[fetch], smoothing_time = smooth(features, adj, 'ap_appro',
                                                     {
                                                        'smooth_alpha': alpha,
                                                        'cache'       : False,
                                                     },fetch=fetch)
        return new_features, smoothing_time

    smoothing_time = 0

    # if model_config['F']:
    #     Y = y_train + y_val + y_test
    #     _, MX, _, _, _ = QM(features.toarray(), Y)
    #     inter_intra_variance(features.toarray(), Y, display=True)
    #     _, MXF, _, _, _ = QM(features.dot(F).toarray(), Y)
    #     inter_intra_variance(features.dot(F).toarray(), Y, display=True)
    #
    #     diff = ((MX-MXF)**2).sum(1)
    #     MX = (MX**2).sum(1)
    #     MXF = (MXF**2).sum(1)
    #     percent = diff/MX
    #     print(np.mean(MX), np.mean(MXF), np.mean(diff), np.mean(percent), np.mean(diff)/np.mean(MX))

    # attribute graph convolution
    if F is not None:
        t = time.time()
        try:
            import cupy as cp
            features = cp.sparse.csc_matrix(features)
            F = cp.sparse.csr_matrix(F)
            features = features.dot(F).todense().get()
        except ImportError:
            features = features.dot(F)
            if sparse.issparse(features):
                features = features.todense()
        smoothing_time += time.time()-t
        if model_config['connection'][0] == 'c': # time optimization for gcn
            features, _ = sample_graph_convolution(features, 1)
            model_config['connection'][0] = 'f'
    features = features.astype(np.float32)

    # sample affinity graph convolution
    if model_config['G']:
        if type(model_config['G']) == int:
            features, t = sample_graph_convolution(features, model_config['G'])
        elif model_config['G'] == 'LP':
            features, t = sample_graph_convolution_lp(features, 0.2)
        else:
            features, t = sample_graph_convolution(features, 2)
    else:
        t = 0
    smoothing_time += t

    # if model_config['F']:
    #     Y = y_train + y_val + y_test
    #     _, MGX, _, _, _ = QM(features, Y)
    #     inter_intra_variance(features, Y, display=True)
    #     _, MGXF, _, _, _ = QM(F.T.dot(features.T).T, Y)
    #     inter_intra_variance(F.T.dot(features.T).T, Y, display=True)
    #
    #     diff = ((MGX-MGXF)**2).sum(1)
    #     MGX = (MGX**2).sum(1)
    #     MGXF = (MGXF**2).sum(1)
    #     percent = diff/MGX
    #     print(np.mean(MGX), np.mean(MGXF), np.mean(diff), np.mean(percent), np.mean(diff)/np.mean(MGX))

    # features = normalize(features, norm='l1', axis=1)
    # features = symmetric_normalize(features)

    if model_config['inter-intra-var']:
        variance = inter_intra_variance(features.toarray() if sparse.issparse(features) else np.array(features),
                                        (y_train + y_val + y_test).astype(np.bool).astype(np.float), display=False)
    else:
        variance = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if model_config['cvr'] is not None:
        if sparse.issparse(features):
            features = features.toarray()
        Y = (y_train + y_val + y_test)
        features = cvr(features, Y, lam=model_config['cvr'])[0].astype(features.dtype)
        # features = cvr(features, y_train, lam=model_config['cvr'])[0].astype(features.dtype)


    if model_config['Model'] == 'LP':
        train_time = time.time()
        test_acc, test_acc_of_class, _ = LP(adj, model_config['alpha'], y_train, y_test)
        train_time = time.time() - train_time
        print("Test set results: accuracy= {:.5f}".format(test_acc))
        print("accuracy of each class=", test_acc_of_class)
        print("Total time={}s".format(time.time()-very_begining))
        return test_acc, 0, train_time, train_time, variance

    support = [preprocess_adj(adj)]
    num_supports = 1

    # Speed up for MLP
    if model_config['connection'] == ['f' for _ in range(len(model_config['connection']))]:
        train_features = features[train_mask]
        y_train = y_train[train_mask].astype(np.int32)
        y_val = y_val[val_mask].astype(np.int32)
        y_test = y_test[test_mask].astype(np.int32)

        val_features = features[val_mask]
        test_features = features[test_mask]
        labels_mask = np.ones(train_mask.sum(), dtype=np.int32)
    else:
        train_features = features
        val_features = features
        test_features = features
        labels_mask = train_mask.astype(np.int32)
        y_train = y_train.astype(np.int32)

    input_dim=features.shape[1]
    if sparse.issparse(features):
        train_features = sparse_to_tuple(train_features)
        val_features = sparse_to_tuple(val_features)
        test_features = sparse_to_tuple(test_features)
        features = sparse_to_tuple(features)

    # Define placeholders
    placeholders = {
        'support': [tf1.sparse_placeholder(tf.float32, name='support' + str(i)) for i in range(num_supports)],
        'features': tf1.sparse_placeholder(tf.float32, name='features') if isinstance(features, tf1.SparseTensorValue) else tf1.placeholder_with_default(train_features, shape=[None, features.shape[1]], name='features'),
        # 'features': tf1.placeholder(shape=[None, features.shape[1]], dtype=tf.float32, name='features'),
        'labels': tf1.placeholder_with_default(y_train, name='labels', shape=(None, y_train.shape[1])),
        'labels_mask': tf1.placeholder_with_default(labels_mask, shape=(None), name='labels_mask'),
        'dropout': tf1.placeholder_with_default(0., name='dropout', shape=()),
        'num_features_nonzero': tf1.placeholder_with_default(train_features[1].shape if isinstance(train_features, tf1.SparseTensorValue) else [0],
                                                            shape=(1), name='num_features_nonzero'),
        'adj_nnz': tf1.placeholder_with_default(support[0].values.shape, shape=(1), name='adj_nnz'),
        'triplet': tf1.placeholder_with_default([[]], name='triplet', shape=(None, None)),
        'noise_sigma': tf1.placeholder_with_default(0., name='noise_sigma', shape=()),
        'training': tf1.placeholder_with_default(False, name='training', shape=())
    }

    # Create model
    model = GCN_MLP(model_config, placeholders, input_dim=input_dim)

    # Random initialize
    sess.run(tf.global_variables_initializer())

    # Construct feed dictionary
    if model_config['connection'] == ['f' for _ in range(len(model_config['connection']))]:
        if isinstance(features, tf1.SparseTensorValue):
            train_feed_dict = {
                placeholders['features'] : train_features,
                placeholders['dropout']  : model_config['dropout'],
                placeholders['training'] : True,
            }
        else:
            train_feed_dict = {
                placeholders['dropout'] : model_config['dropout'],
                placeholders['training'] : True
            }

        valid_feed_dict = construct_feed_dict(
            val_features, support, y_val,
            np.ones(val_mask.sum(), dtype=np.bool), 0, placeholders)

        test_feed_dict = construct_feed_dict(
            test_features, support, y_test,
            np.ones(test_mask.sum(), dtype=np.bool), 0, placeholders)
    else:
        train_feed_dict = construct_feed_dict(train_features, support, y_train, train_mask, model_config['dropout'], placeholders)
        valid_feed_dict = construct_feed_dict(val_features, support, y_val, val_mask, 0, placeholders)
        test_feed_dict = construct_feed_dict(test_features, support, y_test, test_mask, 0, placeholders)

    # Some support variables
    acc_list = []
    max_valid_acc = 0
    max_train_acc = 0
    min_train_loss = 1000000
    t_test = time.time()
    test_cost, test_acc = 0, 0
    valid_loss, valid_acc = 0, 0
    test_duration = time.time() - t_test
    train_time = 0
    step = 1

    def batch_wise_train(features, y, batch_size=1000, training=False):
        loss_acc = []
        rvals = [model.cross_entropy_loss, model.accuracy]
        if training: rvals.append(model.opt_op)
        labels_mask = np.ones(batch_size, dtype=np.int32)
        for i in range(0, features.shape[0], batch_size):
            size = min(batch_size, features.shape[0] - i)
            if labels_mask.shape[0] != size:
                labels_mask = np.ones(size, dtype=np.int32)
            train_loss, train_acc = sess.run(rvals, {
                placeholders['dropout']: model_config['dropout'] if training else 0.,
                placeholders['training']: training,
                placeholders['features']: features[i:i + size],
                placeholders['labels']: y[i:i + size],
                placeholders['labels_mask']: labels_mask,
            })[:2]
            loss_acc.append([train_loss*size, train_acc*size])
        train_loss, train_acc = np.sum(loss_acc, axis=0)/features.shape[0]
        return train_loss, train_acc

    # print(time.time() - very_begining)
    if model_config['train']:
        # test_cost, test_acc = sess.run(
        #     [model.cross_entropy_loss, model.accuracy],
        #     feed_dict=test_feed_dict)
        # valid_loss, valid_acc, valid_summary = sess.run([model.cross_entropy_loss, model.accuracy, model.summary],
        #                                                 feed_dict=valid_feed_dict)

        # Train model
        print('training...')
        for step in range(model_config['epochs']):

            # Training step
            t = time.time()
            # train_loss, train_acc = batch_wise_train(train_features, y_train, training=True)
            sess.run(model.opt_op, feed_dict=train_feed_dict)
            t = time.time()-t
            train_time += t

            # If it's best performence so far, evalue on test set
            # evaluate per 20 steps, or when train almost ends
            if step > model_config['epochs']*0.1 and (step > model_config['epochs']*0.9 or step%20 == 0):
                train_loss, train_acc, train_summary = sess.run([model.cross_entropy_loss, model.accuracy, model.summary],
                                                                feed_dict=train_feed_dict)
                if args.verbose:
                    print(f"Epoch: {step:04d} train_loss= {train_loss:.3f}",
                          f"train_acc= {train_acc:.3f} time= {t:.5f}", end=' ')
                if model_config['validate']:
                    t = time.time()
                    # valid_loss, valid_acc = batch_wise_train(val_features, y_val)
                    valid_loss, valid_acc = sess.run([model.cross_entropy_loss, model.accuracy], feed_dict=valid_feed_dict)
                    t = time.time()-t
                    acc_list.append(valid_acc)
                    if args.verbose:
                        print(f"val_loss= {valid_loss:.3f} val_acc= {valid_acc:.3f} time= {t:.5f}", end=' ')
                    if valid_acc >= max_valid_acc:
                        max_valid_acc = valid_acc

                        t = time.time()
                        # test_cost, test_acc = batch_wise_train(test_features, y_test)
                        test_cost, test_acc = sess.run([model.cross_entropy_loss, model.accuracy], feed_dict=test_feed_dict)
                        t = time.time()-t

                        if args.verbose:
                            print(f"test_loss= {test_cost:.3f} test_acc= {test_acc:.3f} time= {t:.5f}", end=' ')
                    if args.verbose: print()
                else:
                    acc_list.append(train_acc)
                    if train_loss < min_train_loss:
                        min_train_loss = train_loss
                        t_test = time.time()
                        test_cost, test_acc = sess.run(
                            [model.cross_entropy_loss, model.accuracy],
                            feed_dict=test_feed_dict)
                        test_duration = time.time() - t_test

                        if args.verbose:
                            print(f"test_loss= {test_cost:.3f} test_acc= {test_acc:.3f}")

        else:
            print("Optimization Finished!")

        print(f"Test set results: cost= {test_cost:.5f} accuracy= {test_acc:.5f} time= {test_duration:.5f}")

    print(f"Total time={time.time()-very_begining}s")

    return test_acc, train_time/step*1000, smoothing_time, train_time+smoothing_time, variance


if __name__ == '__main__':

    acc = [[] for i in configuration['model_list']]
    duration = [[] for i in configuration['model_list']]
    smoothing_times = [[] for i in configuration['model_list']]
    total_train_times = [[] for i in configuration['model_list']]

    for r in range(configuration['repeating']):
        variances = []
        for model_config, i in zip(configuration['model_list'], range(len(configuration['model_list']))):
            # Set random seed
            seed = model_config['random_seed']
            np.random.seed(seed)
            model_config['random_seed'] = np.random.random_integers(1073741824)

            # Initialize session
            with tf.Graph().as_default():
                tf1.set_random_seed(seed)
                gpu_options = tf1.GPUOptions(allow_growth = True)
                with tf1.Session(config=tf1.ConfigProto(
                        # intra_op_parallelism_threads=model_config['threads'],
                        gpu_options=gpu_options)) as sess:
                    test_acc, t, smoothing_time, \
                        total_train_time, variance = train(model_config, sess)
                    acc[i].append(test_acc)
                    duration[i].append(t)
                    smoothing_times[i].append(smoothing_time)
                    total_train_times[i].append(total_train_time)
                    variances.append([model_config['name']]+variance)
            print('repeated ', r, 'rounds')
        variance_head = ["NAME", "TotalVar", "IntraVar", "InterVar",
                        "Intra/Total", "Inter/Total", "Intra/Inter", "Inter/Intra",
                        "TotalSTD", "IntraSTD", "InterSTD",
                        "IntraSTD/InterSTD", "InterSTD/IntraSTD"]
        if model_config['inter-intra-var']:
            print('\nVARIANCE')
            print(tabulate(variances, headers=variance_head, numalign="left"))

        acc_means = np.mean(acc, axis=1)
        acc_stds = np.std(acc, axis=1)/np.sqrt(configuration['repeating'])
        duration_mean = np.mean(duration, axis=1)
        smoothing_times_mean = np.mean(smoothing_times, axis=1)
        total_train_times_mean = np.mean(total_train_times, axis=1)

        print()
        for line, model_config in zip(acc, configuration['model_list']):
            print(' '.join(f'{j:.6f}' for j in line), model_config['name'])

        print("UTIL REPEAT\t{}".format(r+1))
        print("{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format('DATASET', 'train_size', 'RESULTS', 'STD', 'STEP(ms)', 'SMOOTHING(s)', 'TOTAL_TIME(s)', 'NAME'))
        for model_config, acc_mean, acc_std, t, smoothing_time, total_train_time in zip(configuration['model_list'], acc_means, acc_stds, duration_mean, smoothing_times_mean, total_train_times_mean):
            print(f"{model_config['dataset']:<8}", f"{model_config['train_size']:<8}",
                  f"{acc_mean:<8.6f}", f"{acc_std:<8.6f}",
                  f"{t:<8.2f}", f"{smoothing_time:<8.3f}", f"{total_train_time:<8.3f}",
                  f"{model_config['name']:<8}",
                  sep='\t')
