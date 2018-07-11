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

FB15K237_dir = '/home/haoyu/downloads/FB15K-237/processed'
NELL995_dir = '/home/haoyu/downloads/NELL-995/processed'
WN18RR_dir = '/home/haoyu/downloads/WN18-RR/processed'

def read_embedding(file_dir):
    return np.load(file_dir)

def read_entity2id_inverse(file_dir):
    id2entity = {}
    with open(file_dir, 'rb') as f:
        entity2id = pickle.load(f)
    for k, v in entity2id.items():
        id2entity[v] = k
    return id2entity

def normalize(chi):
    return chi / np.linalg.norm(chi, axis=1, keepdims=True)

def get_k_neighbour(chi, node, k, id2entity):
    chi = normalize(chi)
    sim = np.reshape(chi[node].dot(chi.T), -1)
    sim_nodes = np.argpartition(sim, -k)[-k:]
    return [id2entity[node] for node in sim_nodes]

def visualize(chi_file, entity2id_file, node):
    chi = read_embedding(chi_file)
    id2entity = read_entity2id_inverse(entity2id_file)
    print(id2entity[node])
    print(get_k_neighbour(chi, node, 10, id2entity))


visualize(os.path.join(FB15K237_dir, 'train.npz.chi.npy'), os.path.join(FB15K237_dir, 'entity2id.pkl'), 0)