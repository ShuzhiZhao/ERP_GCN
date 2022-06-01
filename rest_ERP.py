## zhaoshuzhi
import os
import numpy as np
import scipy.io as scio
from scipy.signal import hilbert
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
from utils import *
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_add
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from random import sample
np.seterr(divide='ignore',invalid='ignore')

## Adj_Matrix
def Adj_Matrix(corr_matrix_z):
    Nneighbours=8
    idx = np.argsort(-corr_matrix_z)[:, 1:Nneighbours + 1]
#     print('input corr_matrix:',corr_matrix_z,'\nidx:',idx)
    dist = np.array([corr_matrix_z[i, idx[i]] for i in range(corr_matrix_z.shape[0])])
    dist[dist < 0.1] = 0
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
#     print('dist:\n',dist)
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
    adj_mat_sp = W - W.multiply(bigger) + W.T.multiply(bigger)
    t0 = time.time()
    adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat_sp)
    
    return adj_mat

## Laplacian Matrix: L=I-D(-1/2)AD(-1/2)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def subsample(data,labels,percent):
    sample_list = []
    sample_num = int(percent*len(data))
    sample_list = [i for i in range(len(data))]
    sample_list = sample(sample_list,sample_num)
    data = [data[i] for i in sample_list]
    labels = [labels[i] for i in sample_list]
    
    return data,labels

def cal_MI(data):
    from sklearn.metrics import normalized_mutual_info_score
    varlenDis = np.array(data,dtype=object).shape[0]
    MImatrix = [[0 for col in range(varlenDis)] for row in range(varlenDis)]
    for ii in range(varlenDis):
        for jj in range(varlenDis):
            if ii > jj :
                temp = normalized_mutual_info_score(data[ii],data[jj])
                MImatrix[ii][jj] = temp
                MImatrix[jj][ii] = temp
#     print('MImatrix:\n',np.array(MImatrix,dtype=object).shape,'\n',MImatrix,'\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n')
    return MImatrix

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
    num_layers=3
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    ## 1stGCN
    model_test = ChebNet(block_dura, filters, Nlabels, gcn_layer=num_layers,dropout=0.25,gcn_flag=True)
    model_test = model_test.to(device)
    adj_mat = torch.stack(adj_mat)
    adj_mat = adj_mat.to(device)
    loss_func = nn.CrossEntropyLoss()
    num_epochs=10
    print(model_test)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    model_fit_evaluate(model_test,adj_mat[0],device,train_loader,test_loader,optimizer,loss_func,num_epochs)

    ## ChebNet
    model_test = ChebNet(block_dura, filters, Nlabels, K=3,gcn_layer=num_layers,dropout=0.25)
    model_test = model_test.to(device)
    print(model_test)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    model_fit_evaluate(model_test,adj_mat[0],device,train_loader,test_loader,optimizer,loss_func,num_epochs)
    return acc,loss

def mainFunction(rest_dir,ERP_dir) :
    from sklearn.metrics.pairwise import cosine_similarity
    ## load resting EEG signals of public data
    PCC_corrR = []
    PLV_corrR = []
    cos_corrR = []
    MI_corrR = []
    data_sliR = []  # Resting EEG signal of time slice
    labels_sliR = [] # labels of resting PD
#     rest_dir = '/media/lhj/Momery/PD_GCN/PD_EEGREST'
    files = os.listdir(rest_dir)
    print("++++++++++++++++++++++ load data ... ++++++++++++++++++++++")
    for file in files: 
#         print("++++++++++++++++++++++ The process of"+file+" resting PD ++++++++++++++++++++++")
        data=scio.loadmat(rest_dir+'/'+file) 
        ll = list((data.keys()))
    #     print('data:\n',data['EEG'][0][0]['data'].shape)
        for jj in range(0,80000,700):
            temp = data['EEG'][0][0]['data'][0:64,jj:jj+700]  # normalize(data['EEG'][0][0]['data'][0:64,jj:jj+700], axis=1, norm='l1')
    #         print('resting EEG data of slice:\n',temp.shape)
            if temp.shape[1] == 700:
                data_sliR.append(temp) #.ravel()
        #         print('PCC shape:',np.corrcoef(data['EEG'][0][0]['data'][0:64,jj:jj+700]).shape)
                PCC_corrR.append(np.corrcoef(temp))
        #         print('PLV shape:',np.cov(np.angle(hilbert(data['EEG'][0][0]['data'][0:64,jj:jj+700]),deg=False)).shape)
                PLV_corrR.append(np.cov(np.angle(hilbert(temp))))
#                 print('cos shape:\n',cosine_similarity(temp).shape,'\n',cosine_similarity(temp))
                cos_corrR.append(cosine_similarity(temp))
                MI_corrR.append(np.array(cal_MI(temp)))
                labels_sliR.append([1])            
    ## load ERP data of pitch permutation
    data_triEPD = [] # ERP data of trials
    labels_triEPD = [] # labels of ERPs
#     ERP_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP'
    # PD
    PCC_corrEPD = []
    PLV_corrEPD = []
    cos_corrEPD = []
    MI_corrEPD = []
    files = os.listdir(ERP_dir+'/P40')
    for file in files: 
#         print("++++++++++++++++++++++ The process of"+file+" ERP PD ++++++++++++++++++++++")
        data=scio.loadmat(ERP_dir+'/P40/'+file) 
        keywords = list((data.keys()))
        for i in keywords:
            if 'Category_1' in i:
                temp = data[i][0:64,:] # normalize(data[i][0:64,:], axis=1, norm='l1')
    #             print('data:',data[i][0:64,:])
                data_triEPD.append(temp.tolist()) #.ravel()
                PCC_corrEPD.append(np.corrcoef(temp))
                PLV_corrEPD.append(np.cov(np.angle(hilbert(temp))))
#                 print('cos shape:\n',cosine_similarity(temp).shape,'\n',cosine_similarity(temp))
                cos_corrEPD.append(cosine_similarity(temp))
                MI_corrEPD.append(np.array(cal_MI(temp)))
                labels_triEPD.append([1])
    # HC
    data_triEHC = []
    labels_triEHC = []
    PCC_corrEHC = []
    PLV_corrEHC = []
    cos_corrEHC = []
    MI_corrEHC = []
    files = os.listdir(ERP_dir+'/H40')
    for file in files: 
#         print("++++++++++++++++++++++ The process of"+file+" ERP HC ++++++++++++++++++++++")
        data=scio.loadmat(ERP_dir+'/H40/'+file) 
        keywords = list((data.keys()))
        for i in keywords:
            if 'Category_1' in i:
                temp = data[i][0:64,:] # normalize(data[i][0:64,:], axis=1, norm='l1')
    #             print('data:',data[i][0:64,:])
                data_triEHC.append(temp.tolist())  #.ravel()
                PCC_corrEHC.append(np.corrcoef(temp))
                PLV_corrEHC.append(np.cov(np.angle(hilbert(temp))))
#                 print('cos shape:\n',cosine_similarity(temp).shape,'\n',cosine_similarity(temp))
                cos_corrEHC.append(cosine_similarity(temp))
                MI_corrEHC.append(np.array(cal_MI(temp)))
                labels_triEHC.append([0])
    print("++++++++++++++++++++++ The sum of pulic and our data about PD ++++++++++++++++++++++")
    # print(type(data_sliR),type(labels_sliR),type(data_triE),type(labels_triE))
    print('resting EEG singals:\n',np.array(data_sliR,dtype=object).shape,np.array(labels_sliR,dtype=object).shape,'\nERP data PD:\n',np.array(data_triEPD,dtype=object).shape,np.array(labels_triEPD,dtype=object).shape,'\nERP data HC:\n',np.array(data_triEHC,dtype=object).shape,np.array(labels_triEHC,dtype=object).shape)
    print('PCC resting:',np.array(PCC_corrR).shape,'PLV resting:',np.array(PLV_corrR).shape,'PCC EPD:',np.array(PCC_corrEPD).shape,'PLV EPD:',np.array(PLV_corrEPD).shape,'PCC EHC:',np.array(PCC_corrEHC).shape,'PLV EHC:',np.array(PLV_corrEHC).shape)
    print('MI resting:',np.array(MI_corrR).shape,'MI EHC:',np.array(MI_corrEHC).shape,'MI EPD:',np.array(MI_corrEPD).shape)
    print('cos resting:',np.array(cos_corrR).shape,'cos EHC:',np.array(cos_corrEHC).shape,'cos EPD:',np.array(cos_corrEPD).shape)
    
    ## Preprocessing
    # RPD V.S EHC in MI
    MI_corrR1,labels_sliR1 = subsample(MI_corrR,labels_sliR,0.58)
    corr_Matrix = MI_corrR1+MI_corrEHC
    labels = labels_sliR1+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of resting PD V.S ERP HC in MI ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape:',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of resting PD V.S ERP HC in MI ++++++++++++++++++++++")
    # EPD V.S EHC in MI
    corr_Matrix = MI_corrEPD+MI_corrEHC
    labels = labels_triEPD+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S ERP HC in MI ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape:',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S ERP HC in MI ++++++++++++++++++++++")
    # EPD V.S EHC in MI
    MI_corrR1,labels_sliR1 = subsample(MI_corrR,labels_sliR,0.58)
    corr_Matrix = MI_corrR1+MI_corrEPD
    labels = labels_sliR1+labels_triEPD
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S Resting PD in MI ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape:',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S Resting PD in MI ++++++++++++++++++++++")
    
    # RPD V.S EHC in cos
    cos_corrR1,labels_sliR1 = subsample(cos_corrR,labels_sliR,0.58)
    corr_Matrix = cos_corrR1+cos_corrEHC
    labels = labels_sliR1+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of resting PD V.S ERP HC in cos ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of resting PD V.S ERP HC in cos ++++++++++++++++++++++")
    # EPD V.S EHC in PLV
    corr_Matrix = cos_corrEPD+cos_corrEHC
    labels = labels_triEPD+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S ERP HC in cos ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S ERP HC in cos ++++++++++++++++++++++")
    # EPD V.S RPD in PLV
    cos_corrR1,labels_sliR1 = subsample(cos_corrR,labels_sliR,0.58)
    corr_Matrix = cos_corrR1+cos_corrEPD
    labels = labels_sliR1+labels_triEPD
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S RPD HC in cos ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S RPD HC in cos ++++++++++++++++++++++")
    
    # RPD V.S EHC in PLV
    PLV_corrR1,labels_sliR1 = subsample(PLV_corrR,labels_sliR,0.58)
    corr_Matrix = PLV_corrR1+PCC_corrEHC
    labels = labels_sliR1+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of resting PD V.S ERP HC in PLV ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of resting PD V.S ERP HC in PLV ++++++++++++++++++++++")
    # EPD V.S EHC in PLV
    corr_Matrix = PLV_corrEPD+PCC_corrEHC
    labels = labels_triEPD+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S ERP HC in PLV ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S ERP HC in PLV ++++++++++++++++++++++")
    # EPD V.S RPD in PLV
    PLV_corrR1,labels_sliR1 = subsample(PLV_corrR,labels_sliR,0.58)
    corr_Matrix = PLV_corrR1+PCC_corrEPD
    labels = labels_sliR1+labels_triEPD
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S RPD HC in PLV ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S RPD HC in PLV ++++++++++++++++++++++")
    
    # RPD V.S EHC in PCC
    PCC_corrR1,labels_sliR1 = subsample(PCC_corrR,labels_sliR,0.58)
    corr_Matrix = PCC_corrR1+PCC_corrEHC
    labels = labels_sliR1+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of resting PD V.S ERP HC in PCC ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of resting PD V.S ERP HC in PCC ++++++++++++++++++++++")
    # EPD V.S EHC in PCC
    corr_Matrix = PCC_corrEPD+PCC_corrEHC
    labels = labels_triEPD+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S ERP HC in PCC ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S ERP HC in PCC ++++++++++++++++++++++")
    # EPD V.S RPD in PCC
    PCC_corrR1,labels_sliR1 = subsample(PCC_corrR,labels_sliR,0.5)
    corr_Matrix = PCC_corrR1+PCC_corrEHC
    labels = labels_sliR1+labels_triEHC
    adj_mat = Adj_Matrix(corr_Matrix[1])
    corr_Matrix,labels = subsample(corr_Matrix,labels,0.88)
    print("++++++++++++++++++++++ The Process of ERP PD V.S RPD HC in PCC ++++++++++++++++++++++")
    print('corr_Matrix shape:',type(corr_Matrix),'adj_mat shape:',type(adj_mat),'label shape',type(labels))
    testChebNet(corr_Matrix,[adj_mat],labels)
    print("++++++++++++++++++++++ The End of ERP PD V.S RPD HC in PCC ++++++++++++++++++++++")

        
## dir of data
rest_dir = '/media/lhj/Momery/PD_GCN/PD_EEGREST'
ERP_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP'
mainFunction(rest_dir,ERP_dir)    