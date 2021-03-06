import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import pickle
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import sys
import os

import graphwave as gw
from characteristic_functions import *

TAUS = [1, 10, 25, 50]
ORDER = 11
PROC = 'approximate'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 2

def read_graph(sparse_npz_file):
    sparse_adjacency = scipy.sparse.load_npz(sparse_npz_file)
    G = nx.from_scipy_sparse_matrix(sparse_adjacency)
    print("Finish reading graph from", sparse_npz_file)
    return G

from graphwave import graphwave_alg

def compute_graph_embedding(G, time_pnts, taus,
                            approximate_lambda=True,
                            order=ORDER, proc=PROC, nb_filters=NB_FILTERS):
    chi, heat_print, taus = graphwave_alg(G, time_pnts, taus, verbose=True, approximate_lambda=True,
                            order=ORDER, proc=PROC, nb_filters=NB_FILTERS)
    return chi

def compute_from_sparse_npz(sparse_npz_file, time_pnts=np.linspace(0,100,25),
                            taus='auto', save=True,
                            approximate_lambda=True,
                            order=ORDER, proc=PROC, nb_filters=NB_FILTERS,
                            **kwargs
                            ):
    G = read_graph(sparse_npz_file)
    chi = compute_graph_embedding(G, time_pnts, taus,
                                  approximate_lambda=approximate_lambda,
                                  order=order, proc=proc, nb_filters=nb_filters)
    if save:
        np.save(sparse_npz_file + '.chi', chi)
    return chi

FB15K237_dir = '/home/haoyu/downloads/FB15K-237/processed'
NELL995_dir = '/home/haoyu/downloads/NELL-995/processed'
WN18RR_dir = '/home/haoyu/downloads/WN18-RR/processed'

compute_from_sparse_npz(os.path.join(FB15K237_dir, 'train.npz'))
#compute_from_sparse_npz(os.path.join(FB15K237_dir, 'valid.npz'))
#compute_from_sparse_npz(os.path.join(FB15K237_dir, 'test.npz'))
#compute_from_sparse_npz(os.path.join(NELL995_dir, 'raw.npz'))
#compute_from_sparse_npz(os.path.join(WN18RR_dir, 'train.npz'))
#compute_from_sparse_npz(os.path.join(WN18RR_dir, 'valid.npz'))
#compute_from_sparse_npz(os.path.join(WN18RR_dir, 'test.npz'))