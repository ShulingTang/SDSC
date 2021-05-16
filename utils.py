import numpy as np

import scipy.io as sio

def process_data(args):
    if args.db == 'coil20':
        # load data
        data = sio.loadmat('datasets/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 40
        weight_coef = 1.0
        weight_selfExp = 75
        weight_cc = 1.0
        kmeansNum = 120
        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
        SC_method = False # accelerate
        # warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")
    elif args.db == 'Orl':
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 40
        weight_coef = 1.0
        weight_selfExp = 0.1
        weight_cc = 1.0
        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
        kmeansNum = 10
        SC_method = True
    elif args.db == 'Coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 60
        weight_coef = 1.0
        weight_selfExp = 15
        kmeansNum = 75
        weight_cc = 1.0
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8
        SC_method = True
    elif args.db == 'Mnist10000':
        # load data
        data = sio.loadmat('datasets/Mnist10000.mat')
        x, y = data['X'].reshape((-1, 1, 28, 28)), data['y']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
        kmeansNum = 62
        channels = [1, 15]
        kernels = [3, 3]
        epochs = 100
        weight_coef = 1.0
        weight_selfExp = 15
        weight_cc = 1.0
        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8
        SC_method = True
    elif args.db == 'Mnist20000':
        # load data
        data = sio.loadmat('datasets/Mnist20000.mat')
        x, y = data['X'].reshape((-1, 1, 28, 28)), data['y']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        kmeansNum = 100
        channels = [1, 15]
        kernels = [3, 3]
        epochs = 100
        weight_coef = 1.0
        weight_selfExp = 15
        weight_cc = 1.0

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
        SC_method =True
    return x, y, num_sample, kmeansNum, channels, kernels, epochs, weight_coef, weight_selfExp, \
           alpha, dim_subspace, weight_cc, ro, SC_method