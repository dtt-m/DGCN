import argparse
import math, scipy.stats as st
# from matplotlib import axis
import numpy as np

import random
import numpy as np
import torch


import networkx as nx

from collections import Counter
from ordered_set import OrderedSet

from collections import defaultdict as ddict

import os, logging, tqdm
from sklearn.cluster import KMeans
import datetime

import torch.nn.functional as F

neg_num_samp = 10


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
    parser.add_argument('--model', type=str, default='', help='model')
    parser.add_argument('--task', type=str, default='link', help='task: node, link')

    parser.add_argument('--ds_path', type=str, default='Datasets')
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--custom', type=str, default='')

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--drop', default=0.1, type=float)
    parser.add_argument('--decay', default=1e-04, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0.1, type=float, help='weight decay')

    parser.add_argument('--nhid', type=int, default=256, help='hidden size')

    parser.add_argument('--n_runs', type=int, default=5, help='batch size')
    parser.add_argument('--batch', type=int, default=1024, help='batch size')

    parser.add_argument('--epoch', type=int, default=1000, help='num of iteration')
    parser.add_argument('--d_epoch', type=int, default=100, help='diffusion of iteration')
    parser.add_argument('--timesteps', type=int, default=100, help='diffusion of iteration')
    parser.add_argument('--pre_step', type=int, default=2000, help='pretrained iteration')

    parser.add_argument('--patience', type=int, default=50, help='early stopping') #node patience: 10

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=bool, default=False, help='use GPU')
    parser.add_argument('--gpu', type=int, default=3, help='gpu id')

    parser.add_argument('--no_diff', action='store_true') 
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--k', type=int, default=5, help='Number of hops for neighbors')
    parser.add_argument('--num_neighbors', type=int, default=20, help='Number for neighbors')
    parser.add_argument('--cl1', type=float, default=0.01,
                       help='Weight for 1-hop contrastive loss')
    parser.add_argument('--cl2', type=float, default=0.01,
                       help='Weight for k-hop contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for contrastive loss')
    #parser.add_argument('--workers', type=int, default=10, help='Number of processes to construct batches')

    return parser.parse_args() 




def prepare_saved_path(args):

    # dataset folder
    save_path = os.path.join(args.save_path, args.dataset)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # model folder index
    now = datetime.datetime.now()
    #index = len(next(os.walk(save_path))[1])

    save_folder =  '_'.join([str(now.day), str(now.month), str(now.strftime("%H:%M:%S"))])
    save_path = os.path.join(save_path, save_folder)
    
    os.mkdir(save_path)

    #
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        for k,v in vars(args).items():
            f.write(str(k) + ': ' + str(v) + '\n')

    return save_path

def normalize_adj(adj, norm_type=1, iden=False):

    
    if isinstance(adj, np.matrix):
        adj = np.array(adj)
    adj = adj.copy()

    
    if iden:
        adj = adj + np.eye(adj.shape[0])

    epsilon = 1e-10  

    if norm_type == 1:
        
        D = np.sum(adj, axis=1)
        D = np.where(D > 0, D, 1.0)  
        adj_norm = adj / (D.reshape(-1, 1) + epsilon)
    else:
        
        adj_binary = np.where(adj > 0, 1.0, 0.0)  
        degrees = np.sum(adj_binary, axis=1)
        degrees = np.where(degrees > 0, degrees, 1.0)
        D_sqrt = np.diag(1.0 / np.sqrt(degrees + epsilon))
        adj_norm = D_sqrt @ adj_binary @ D_sqrt

    
    adj_norm = np.nan_to_num(adj_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return adj_norm



def negative_sampling(edges, adj, adj1=None, bias=False):

    samples = []
    nei_2_dict = {}
    #2-hops

    for i in range(adj.shape[0]):
        nei_1 = adj[i]
        nei_2 = adj * nei_1.T
        nei_2 = torch.sum(nei_2, axis=1)
        nei_2_dict[i] = torch.where(nei_2 > 0)[0]
        
    for edge in edges:
        neg = np.random.choice(nei_2_dict[edge[0]], neg_num_samp)
        samples.append(neg)

    return np.asarray(samples)


def create_train_degree(data, adj, adj1, path, num=0, name='', data_list=''):
    np.random.seed(0)
    count = 0

    name_file = os.path.join(path, name, 'train_degree' + data_list)
    edges = data['train'][:,:2]

    adj1 = torch.FloatTensor(adj1)
    deg = torch.sum(adj1, 1) ** 0.75
    prob = deg / torch.sum(deg)
    prob = prob.numpy()
    
    samples = []
    for edge in edges:
        neg = np.random.choice(num, neg_num_samp, p=prob)
        samples.append(neg)
    samples = np.asarray(samples)

    triples = np.concatenate((edges, samples), axis=1)
   

    with open(name_file,'w')  as f:
        for t in triples:
            tmp = " ".join([str(j) for j in t])
            f.write(tmp + '\n')
    return


# for link prediction
def create_train_test_split(data, path, num=0, name='', data_list=''):
    np.random.seed(0)
    count = 0
    for split in ['train', 'test', 'val']:
        name_file = os.path.join(path, name, split + data_list)
        edges = data[split]

        # for uniform sampling
        samples = np.random.randint(num, size=(edges.shape[0],neg_num_samp))
        
        # 2-hop neighbors
        #samples = negative_sampling(edges, adj)
        triples = np.concatenate((edges, samples), axis=1)

        with open(name_file,'w')  as f:
            for t in triples:
                tmp = " ".join([str(j) for j in t])
                f.write(tmp + '\n')
        
    return



def create_edge_list(data, path, n_node):

    edges = data[:,:2]

    with open(path, 'w') as f:
        for e in edges:
            f.write(str(e[0]) + '\t' + str(e[1]) + '\t1''\n') #  + 
        for n in range(n_node):
            f.write(str(n) + '\t' + str(n) + '\t1''\n')
    return
def normalize_adj(adj):
    """对称归一化邻接矩阵"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
