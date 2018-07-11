# Processing the entity - relation - entity triples
# 1. Encoding the entities and relations into numbers
# 2. Create Sparse Matrix Graph
# 3. Dump Entity2id, Relation2id

import pickle
import scipy
from scipy import sparse
import numpy as np
import os
from collections import Counter


def build_id_dict(file_list):
    raw = ''
    for file in file_list:
        with open(file) as f:
            raw += f.read()
    triples = raw.split('\n')[:-1]
    entities = []
    relations = []
    for triple in triples:
        head, relation, tail = triple.split()
        entities.append(head)
        entities.append(tail)
        relations.append(relation)
    entity2id, relation2id = {}, {}
    for i, entity in enumerate(Counter(entities)):
        entity2id[entity] = i
    for i, relation in enumerate(Counter(relations)):
        relation2id[relation] = i
    return entity2id, relation2id


def read_file(file_list, entity2id, relation2id):
    raw = ''
    for file in file_list:
        with open(file) as f:
            raw += f.read()
    triples = raw.split('\n')[:-1]
    head_ids, relation_ids, tail_ids = [], [], []
    for triple in triples:
        head, relation, tail = triple.split()
        head_id = entity2id[head]
        tail_id = entity2id[tail]
        relation_id = relation2id[relation]
        head_ids.append(head_id)
        tail_ids.append(tail_id)
        relation_ids.append(relation_id)
    return head_ids, relation_ids, tail_ids


def build_binary_graph(head_ids, tail_ids, num_nodes):
    # head_ids + tail_ids:
    # symmetric matrix, head -> tail = tail -> head
    return scipy.sparse.csc_matrix(([1]*len(head_ids+tail_ids), (head_ids+tail_ids, tail_ids+head_ids)), shape=(num_nodes, num_nodes))

def build_binary_graph_from_file(file_list, entity2id, relation2id, num_nodes):
    head_ids, relation_ids, tail_ids = read_file(file_list, entity2id, relation2id)
    binary_graph = build_binary_graph(head_ids, tail_ids, num_nodes)
    return binary_graph


FB15K237_dir = '/home/haoyu/downloads/FB15K-237'
NELL995_dir = '/home/haoyu/downloads/NELL-995'
WN18RR_dir = '/home/haoyu/downloads/WN18-RR'

def preprocess(file_dir):
    print("Txts in ", file_dir, "are processed.")
    entity2id, relation2id = build_id_dict([os.path.join(file_dir, 'train.txt'),
                                                  os.path.join(file_dir, 'valid.txt'),
                                                  os.path.join(file_dir, 'test.txt')])
    output_dir = os.path.join(file_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'entity2id.pkl'), 'wb') as f:
        pickle.dump(entity2id, f)
    with open(os.path.join(output_dir, 'relation2id.pkl'), 'wb') as f:
        pickle.dump(relation2id, f)
    print("Entity2id and Relation2id have been dumped to pickles.")
    num_nodes = len(entity2id.keys())
    num_relations = len(relation2id.keys())
    print("The number of nodes is", num_nodes)
    print("The number of relations is", num_relations)
    train_graph = build_binary_graph_from_file([os.path.join(file_dir, 'train.txt')], entity2id, relation2id,
                                               num_nodes)
    valid_graph = build_binary_graph_from_file([os.path.join(file_dir, 'valid.txt')], entity2id, relation2id,
                                               num_nodes)
    test_graph = build_binary_graph_from_file([os.path.join(file_dir, 'test.txt')], entity2id, relation2id,
                                               num_nodes)
    scipy.sparse.save_npz(os.path.join(output_dir, 'train.npz'), train_graph)
    scipy.sparse.save_npz(os.path.join(output_dir, 'valid.npz'), valid_graph)
    scipy.sparse.save_npz(os.path.join(output_dir, 'test.npz'), test_graph)

def preprocess_raw(file_dir):
    print("Txts in ", file_dir, "are processed.")
    entity2id, relation2id = build_id_dict([os.path.join(file_dir, 'raw.kb')])
    output_dir = os.path.join(file_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'entity2id.pkl'), 'wb') as f:
        pickle.dump(entity2id, f)
    with open(os.path.join(output_dir, 'relation2id.pkl'), 'wb') as f:
        pickle.dump(relation2id, f)
    print("Entity2id and Relation2id have been dumped to pickles.")
    num_nodes = len(entity2id.keys())
    num_relations = len(relation2id.keys())
    print("The number of nodes is", num_nodes)
    print("The number of relations is", num_relations)
    train_graph = build_binary_graph_from_file([os.path.join(file_dir, 'raw.kb')], entity2id, relation2id,
                                               num_nodes)
    scipy.sparse.save_npz(os.path.join(output_dir, 'raw.npz'), train_graph)

preprocess(FB15K237_dir)
preprocess_raw(NELL995_dir)
preprocess(WN18RR_dir)