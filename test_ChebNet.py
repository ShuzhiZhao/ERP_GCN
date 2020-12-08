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
def build_adjacency(lmdb_dir, name_dir,subname):
    sub_name = []  ## all subjects
    subTrials = []  ## samples
    adj_lst = []  ## adjacency matrix
    lap_lst = []  ## Laplacian eigenvalues
    W = []  # w = eigenvalues, v = eigenvectors
    V = []
    save_dir = './GCN_data/'
    mkdir(save_dir)
    with open(name_dir, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
                pass
            for i in lines.split():
                sub_name.append(i) 
            pass 
#     print(sub_name)
#     for i in range(len(sub_name)):
        ## get all PCC trials from all subjects
        lmdb_env = lmdb.open(lmdb_dir,readonly=True)
        with lmdb_env.begin() as lmdb_mod_txn:
            mod_cursor = lmdb_mod_txn.cursor()
            for idx,(key, value) in enumerate(mod_cursor):
                key = str(key, encoding='utf-8')
                lst = list()
                lst.append(key)
    #             print(lst)
                for i in lst:
                    if subname+'_PCC_' in i:
                        corr_matrix_z = lmdb_display(lmdb_dir,i)
                        corr_matrix_z = sim_adj(corr_matrix_z, i, lmdb_dir) ## consider the difference of trials and sub
                        subTrials.append(i)
                        num_nodes = corr_matrix_z.shape[0]
#                         print('================',i,'=================')
#                         print(channel_Matrix)                
                        Nneighbours=8
                        idx = np.argsort(-corr_matrix_z)[:, 1:Nneighbours + 1]
                        dist = np.array([corr_matrix_z[i, idx[i]] for i in range(corr_matrix_z.shape[0])])
                        dist[dist < 0.1] = 0
                        adj_mat_sp = kNN(dist, idx)   ## scipy.sparse.csr.csr_matrix
#                         adj_narray = adj_mat_sp.toarray().tolist()
#                         adj_lst.append(adj_narray)
#                         sparse.save_npz(save_dir+i+'_adj_mat_sp.npz', adj_mat_sp)
#                         sp_adj = sparse.load_npz(save_dir+i+'_adj_mat_sp.npz')
#                         print('================adj_mat_sp***',i,'=================')
#                         print(np.array(adj_lst).shape)
#                         print(sp_adj) 
                        
                        t0 = time.time()
                        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat_sp)
#                         print('adj_mat ',adj_mat)
                        adj_lst.append(adj_mat)
#                         print('==========================================================')
#                         print('adj_lst ',adj_lst)
                        edge_index = adj_mat._indices()
                        edge_weight = adj_mat._values()
                        row, col = edge_index
                        #degree-matrix
                        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
                        # Compute normalized and rescaled Laplacian.
                        deg = deg.pow(-0.5)
                        deg[torch.isinf(deg)] = 0
                        lap = deg[row] * edge_weight * deg[col]
                        ###Rescale the Laplacian eigenvalues in [-1, 1]
                        ##rescale: 2L/lmax-I; lmax=1.0
                        fill_value = 1  ##-0.5
                        edge_index, lap = add_self_loops(edge_index, -lap, fill_value, num_nodes)
                        laplacian_matrix = sparse.coo_matrix((lap.numpy(),edge_index),shape=(num_nodes,num_nodes))
                        laplacian_narray = laplacian_matrix.toarray().tolist()
                        lap_lst.append(laplacian_narray)
#                         sparse.save_npz(save_dir+i+'_laplacian_matrix.npz', laplacian_matrix)
#                         print('================laplacian_matrix ***',i,'=================')
#                         print(np.array(lap_lst).shape)
#                         print(laplacian_matrix)
                                                
                        w, v = eigh(laplacian_matrix.todense()) # w = eigenvalues, v = eigenvectors numpy.ndarray
                        W.append(w.tolist())
                        V.append(v.tolist())
#                         sparse.save_npz(save_dir+i+'_eigenvalues.npz', w)
#                         sparse.save_npz(save_dir+i+'_eigenvectors.npz', v)
                        K_eigbasis = min(4,num_nodes)
#                         print('================K_eigbasis***',i,'=================')
#                         print(np.array(W).shape,'\n================================\n',np.array(V).shape)
#                         print(w,w.shape,'\n-------------------------\n',v,v.shape)                     
                         
    
    return sub_name,subTrials,adj_lst,lap_lst,W,V 

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

## get ERP matrix(700*64) of someone
## input:mat_dir,subname   out:store matrix in lmdb
def ERP_matrix(ERP_dir,lmdb_dir,subname):
    ERP_Matrix = []
    ERP_lmdb_dir = lmdb_dir+'/ERP_Matrix'
    mkdir(ERP_lmdb_dir)
    ## read all name
#     H40_ERP_dir = ERP_dir+'/H40/'
#     P40_ERP_dir = ERP_dir+'/P40/'
#     H40_files = os.listdir(H40_ERP_dir)
#     P40_files = os.listdir(P40_ERP_dir)
#     for file in H40_files: 
#         print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#         data=scio.loadmat(H40_ERP_dir+file)
#         ll = list((data.keys()))
#         for i in ll:
#             if 'Category' in i:               
#                 ERP_name = file.replace('.mat','')+'_ERP_Matrix_'+i
#                 ## store ERP_matrix
#                 lmdb_env = lmdb.open(ERP_lmdb_dir,map_size = int(1e12)*2)
#                 with lmdb_env.begin(write=True) as lmdb_txn:
#                     lmdb_txn.put(ERP_name.encode(),np.ascontiguousarray(data[i]))
#                     ## del ERP_Matrix of nan
#                     for j in range(len(data[i])):
#                         if 'nan' in str(data[i][j]):
#                             print('the position of nan lie in ', i, '\n', channel_Matrix)
#                             lmdb_del_txn.delete(ERP_name.encode())
                                        
#     for file in H40_files: 
#         print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#         data=scio.loadmat(H40_ERP_dir+file)
#         ll = list((data.keys()))
#         for i in ll:
#             if 'Category' in i:               
#                 ERP_name = file.replace('.mat','')+'_ERP_Matrix_'+i
#                 ## store ERP_matrix
#                 lmdb_env = lmdb.open(ERP_lmdb_dir,map_size = int(1e12)*2)
#                 with lmdb_env.begin(write=True) as lmdb_txn:
#                     lmdb_txn.put(ERP_name.encode(),np.ascontiguousarray(data[i])) 
#                     ## del ERP_Matrix of nan
#                     for j in range(len(data[i])):
#                         if 'nan' in str(data[i][j]):
#                             print('the position of nan lie in ', i, '\n', channel_Matrix)
#                             lmdb_del_txn.delete(ERP_name.encode())
    
    ## get ERP_matrix of subname
    subTrial = []
    lmdb_env = lmdb.open(ERP_lmdb_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            lst = list()
            lst.append(key)
            for i in lst:
                if 'H1_ERP_Matrix_' in i or 'P1_ERP_Matrix_' in i:
                    value=lmdb_mod_txn.get(i.encode())
#                     print(i)
                    Matrix = np.frombuffer(value,dtype=np.float64).reshape(65,700).transpose((1,0))[:,:64]
                    ERP_Matrix.append(Matrix)
                    subTrial.append(i)
    print('ERP_Matrix shape {}'.format(np.array(ERP_Matrix).shape))
    return np.array(ERP_Matrix),subTrial

## main
lmdb_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/test_result/train'
ERP_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP'
ERP_lmdb_dir = '/media/lhj/Momery/PD_GCN/Script/test_ChebNet'
gcDa_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/test_result/gcnData'
sub_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/subject.txt'
subname = 'H10'


sub_name,subTrials,adj_mat,lap_lst,W,V = build_adjacency(lmdb_dir, sub_dir,subname)
labels_lst = subLabels(subTrials)
ERP_Matrix,subTrials1 = ERP_matrix(ERP_dir, ERP_lmdb_dir, subname)
subTrial = subLabels(subTrials1)

###split the entire dataset into train and test tests
###############################
task_contrasts = {"HC": "Healthy_Person",
                  "PD":"Parkinson_Person"
                 }
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 1}

target_name = (list(task_contrasts.values()))
Nlabels = len(target_name)
print('Nlabels {}'.format(Nlabels))

Region_Num = np.array(ERP_Matrix)[0].shape[-1]
print(Region_Num)
block_dura = 700    
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
print(type(label_data_train),'\n',label_data_train)
# print('fmri_data_train ',type(fmri_data_train),'\n','label_data_train ',type(label_data_train))
# train_data = TensorDataset(torch.Tensor(fmri_data_train,dtype=torch.long),torch.Tensor(label_data_train,dtype=torch.long))
# train_sampler = RandomSampler(train_data)
# train_loader = DataLoader(train_data,sampler=train_sampler,**params)
# fmri_train_dataset = HCP_taskfmri_matrix_datasets(fmri_data_train, label_data_train, target_name, block_dura=17, isTrain='train')
ERP_train_dataset = ERP_matrix_datasets(fmri_data_train, label_data_train, target_name, block_dura=700, isTrain='train')
train_loader = DataLoader(ERP_train_dataset, collate_fn=ERP_samples_collate_fn, **params)

####test set
fmri_data_test = [ERP_Matrix[i] for i in test_sid]
label_data_test = pd.DataFrame(np.array([subTrial[i] for i in test_sid]))
print(type(label_data_test),'\n',label_data_test)
# test_data = TensorDataset(torch.Tensor(fmri_data_test,dtype=torch.long),torch.Tensor(label_data_test,dtype=torch.long))
# test_sampler = RandomSampler(test_data)
# test_loader = DataLoader(test_data,sampler=test_sampler,**params)
# fmri_test_dataset = HCP_taskfmri_matrix_datasets(fmri_data_test, label_data_test, target_name, block_dura=17, isTrain='test')
ERP_test_dataset = ERP_matrix_datasets(fmri_data_test, label_data_test, target_name, block_dura=700, isTrain='test')
test_loader = DataLoader(ERP_test_dataset, collate_fn=ERP_samples_collate_fn, **params)



## ChebNet
from model import ChebNet
from model import count_parameters, model_fit_evaluate

filters=32
num_layers=2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
## 1stGCN
model_test = ChebNet(block_dura, filters, Nlabels, gcn_layer=num_layers,dropout=0.25,gcn_flag=True)
#model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)
model_test = model_test.to(device)
# print('type of adj ',type(adj_mat))
# adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
# adj_mat = torch.FloatTensor(adj_mat)
# print(adj_mat,'\n',type(adj_mat),np.array(adj_mat).shape)
# print('adj_mat of stack ',adj_mat,np.array(adj_mat).shape)
adj_mat = torch.stack(adj_mat)
# adj_mat = torch.stack([torch.Tensor(adj_mat[ii]) for ii in range(len(adj_mat))])
adj_mat = adj_mat.to(device)
loss_func = nn.CrossEntropyLoss()
num_epochs=10
print(model_test)
print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
model_fit_evaluate(model_test,adj_mat[1],device,train_loader,test_loader,optimizer,loss_func,num_epochs)


## ChebNet
model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)
model_test = model_test.to(device)
print(model_test)
print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
model_fit_evaluate(model_test,adj_mat[1],device,train_loader,test_loader,optimizer,loss_func,num_epochs)