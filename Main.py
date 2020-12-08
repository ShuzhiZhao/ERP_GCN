from numpy import linalg as LA
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
import os
import lmdb
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
from utils import *
    
## get ERP matrix(700*64) of someone
## input:mat_dir,subname   out:store matrix in lmdb
def ERP_matrix(ERP_dir,subname):
    ERP_Matrix = []
    subTrial = []
    lmdb_env = lmdb.open(ERP_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            lst = list()
            lst.append(key)
            for i in lst:
                for j in subname:
                    if j+'_PCC_' in i:
                        value=lmdb_mod_txn.get(i.encode())
                        Matrix = np.frombuffer(value,dtype=np.float64).reshape(64,64).transpose((1,0))[:,:64]
#                         print(i)
                        ERP_Matrix.append(Matrix)
                        subTrial.append(i)
    print('ERP_Matrix shape {}'.format(np.array(ERP_Matrix).shape))
    return np.array(ERP_Matrix),subTrial
## find channel matrix from lmdb by key
def lmdb_display(env_dir,key):
    import lmdb
    lmdb_env = lmdb.open(env_dir,readonly=True)
    lmdb_txn = lmdb_env.begin()
    value = lmdb_txn.get(key.encode())
#     print(value)
    channel_Matrix = np.frombuffer(value,dtype=np.float64)
#     print(channel_Matrix)
#     print('++++++++++++++++++++++++++++++value++++++++++++++++++++++++++++++++')
    channel_Matrix = channel_Matrix.reshape(64,64)
#     print(channel_Matrix)
    return channel_Matrix

def kNN(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W

## Laplacian Matrix: L=I-D(-1/2)AD(-1/2)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

##define mkdir
def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")    
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+"success")
        return True
    else:
        print(path+"already exist")
        return False

## consider trials and subs for GCN
def sim_adj(corr_matrix_z, sub_name, lmdb_dir):
    import re
    from numpy import linalg as LA
    sim_adj = []
    sim_sub = [[0 for k in range(64)] for i in range(64)]
    name = re.findall(r"(.+?)_PCC", sub_name)
    name = ''.join(name)
#     print('sub_name ', name)
    lmdb_env = lmdb.open(lmdb_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            lst = list()
            lst.append(key)
    #        print(lst)
            for j in lst:
                if name in j and 'Sim' in j:
                    sim_sub += lmdb_display(lmdb_dir,j)
#                     print(sim_sub)
                    sim_sub = (sim_sub-np.mean(sim_sub,axis=0))/np.std(sim_sub,axis=0)
    corr_matrix_z = corr_matrix_z.tolist()
    for i in range(64):
        for k in range(64):
            if i==k:
                corr_matrix_z[i][k] = 0
    corr_matrix_z = np.array(corr_matrix_z)                        
    sim_adj = LA.norm(sim_sub,1)*corr_matrix_z 
#     sim_adj = (sim_adj-np.mean(sim_adj,axis=0))/np.std(sim_adj,axis=0)
#     print('==========================sim_sub '+name+'================================') 
#     print(np.array(sim_sub).shape)
#     print(sim_sub)
#     print('==========================corr_matrix_z '+name+'================================') 
#     print(np.array(corr_matrix_z).shape)
#     print(corr_matrix_z)
#     print('==========================sim_adj '+name+'================================') 
#     print(np.array(sim_adj).shape)
#     print(sim_adj)    
    return sim_adj
    
## build adjacency matrix of KNN from lmdb dataset
## restore in gcn_data
def build_adjacency(lmdb_dir,subname):
#     subTrials = []  ## samples
    adj_mat = []  ## adjacency matrix
#     lmdb_env = lmdb.open(lmdb_dir,readonly=True)
#     with lmdb_env.begin() as lmdb_mod_txn:
#         mod_cursor = lmdb_mod_txn.cursor()
#         for idx,(key, value) in enumerate(mod_cursor):
#             key = str(key, encoding='utf-8')
#             lst = list()
#             lst.append(key)
#             for i in lst:
#                 for j in subname:
#                     if j+'_PCC_' in i:
#                         corr_matrix_z = lmdb_display(lmdb_dir,i)
#                         corr_matrix_z = sim_adj(corr_matrix_z, i, lmdb_dir) ## consider the difference of trials and sub
#                         subTrials.append(corr_matrix_z)
#     meanChannel = [[0 for k in range(64)] for i in range(64)]
#     sumNumber = len(subTrials)
#     for ii in subTrials:
#         meanChannel += ii
#     meanChannel = np.array(meanChannel)/sumNumber
#     print('meanChannel {}\n============\n{}'.format(meanChannel.shape,meanChannel))
#     scio.savemat(lmdb_dir+'/meanChannel.mat',{'meanChannel':meanChannel})
    meanChannel = scio.loadmat(lmdb_dir+'/meanChannel.mat')['meanChannel']
    adj_mat = Lap_sparse(meanChannel)
    return adj_mat

def Lap_sparse(corr_matrix_z):
    Lap_Sparse = []
    num_nodes = corr_matrix_z.shape[0]
    Nneighbours=8
    idx = np.argsort(-corr_matrix_z)[:, 1:Nneighbours + 1]
    dist = np.array([corr_matrix_z[i, idx[i]] for i in range(corr_matrix_z.shape[0])])
    dist[dist < 0.1] = 0
    adj_mat_sp = kNN(dist, idx)   ## scipy.sparse.csr.csr_matrix 
    adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat_sp)
    Lap_Sparse.append(adj_mat)
    
    return Lap_Sparse

def subLabels(subTrials):
    labels_lst = []
    for st in subTrials:
        if st.startswith('H',0,3):
            a=0
            labels_lst.append(a) ## healthy person
        elif st.startswith('P',0,3):
            a=1
            labels_lst.append(a) ## parkinson
        else:
            print('can not define label, label not exit')
#         print('----------------------------',st,'----------------------')
#         print('label: ',labels_lst[subTrials.index(st)],'\nindex: ',subTrials.index(st))
    
#     print('----------------------------',subTrials[0],'----------------------')
#     print('label: ',labels_lst[0])
    
    return labels_lst

## test_ChebNet
def testChebNet(ERP_Matrix,adj_mat,subTrial):
    acc = []
    loss = []
    task_contrasts = {"HC": "Healthy_Person",
                  "PD":"Parkinson_Person"
                 }
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 1}

    target_name = (list(task_contrasts.values()))
    Nlabels = len(target_name)
    Region_Num = np.array(ERP_Matrix)[0].shape[-1]
    print(Region_Num)
    block_dura = 64    
    test_size = 0.2
    randomseed=1234
    test_sub_num = len(ERP_Matrix)
    print('test_sub_num ',test_sub_num)
    rs = np.random.RandomState(randomseed)
    train_sid, test_sid = train_test_split(range(test_sub_num), test_size=test_size, random_state=rs, shuffle=True)
    print('training on %d subjects, validating on %d subjects' % (len(train_sid), len(test_sid)))
    
    ####train set 
    fmri_data_train = [ERP_Matrix[i] for i in train_sid]
    label_data_train = pd.DataFrame(np.array([subTrial[i] for i in train_sid]))
#     print(type(label_data_train),'\n',label_data_train)
    ERP_train_dataset = ERP_matrix_datasets(fmri_data_train, label_data_train, target_name, block_dura=700, isTrain='train')
    train_loader = DataLoader(ERP_train_dataset, collate_fn=ERP_samples_collate_fn, **params)

    ####test set
    fmri_data_test = [ERP_Matrix[i] for i in test_sid]
    label_data_test = pd.DataFrame(np.array([subTrial[i] for i in test_sid]))
#     print(type(label_data_test),'\n',label_data_test)
    ERP_test_dataset = ERP_matrix_datasets(fmri_data_test, label_data_test, target_name, block_dura=700, isTrain='test')
    test_loader = DataLoader(ERP_test_dataset, collate_fn=ERP_samples_collate_fn, **params)

    ## ChebNet
    from model import ChebNet
    from model import count_parameters, model_fit_evaluate

    filters=32
    num_layers=2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## 1stGCN
    model_test = ChebNet(block_dura, filters, Nlabels, gcn_layer=num_layers,dropout=0.25,gcn_flag=True)
    model_test = model_test.to(device)
    adj_mat = torch.stack(adj_mat)
    adj_mat = adj_mat.to(device)
    loss_func = nn.CrossEntropyLoss()
    num_epochs=15
    print(model_test)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    model_fit_evaluate(model_test,adj_mat[0],device,train_loader,test_loader,optimizer,loss_func,num_epochs)

    ## ChebNet
    model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)
    model_test = model_test.to(device)
    print(model_test)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    model_fit_evaluate(model_test,adj_mat[0],device,train_loader,test_loader,optimizer,loss_func,num_epochs)
    return acc,loss

## Look for the problem of subjects in ChebNet
## subname
## subname+'_ERP_Matrix_' in i or ...
def mainChebNet(work_dir):
    ## work_dir including Channel Matrix data, ERP Matrix data,subject.txt
    Channel_dir = work_dir+'/train'
    ERP_dir = work_dir+'/ERP_Matrix'
    sub_dir = work_dir+'/subject.txt'
    ## get all subNames
    sub_name = []
    with open(sub_dir, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
                pass
            for i in lines.split():
                sub_name.append(i) 
            pass 
#     print(sub_name)
    
    ## test ChebNet for Multi-Sub
    ## Channel data
    adj_mat = build_adjacency(Channel_dir,sub_name)
    # ERP data && Label
    ERP_Matrix,ERP_name = ERP_matrix(Channel_dir,sub_name)
    subLabel = subLabels(ERP_name)
    testChebNet(ERP_Matrix,adj_mat,subLabel)
    
    return sub_name

## main program
work_dir = '/media/lhj/Momery/PD_GCN/Script/test_ChebNet'
mainChebNet(work_dir)