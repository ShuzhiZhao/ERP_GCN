## the donate of every and total channel among trials and between subjects
from numpy import linalg as LA
import numpy as np
import scipy.io as scio
import os
import lmdb
import math
from scipy.spatial.distance import cdist

## find channel matrix from lmdb by key
def lmdb_display(env_dir,key):
    import lmdb
    lmdb_env = lmdb.open(env_dir,readonly=True)
    lmdb_txn = lmdb_env.begin()
    value = lmdb_txn.get(key.encode())
#     print(value)
    channel_Matrix = np.fromstring(value,dtype=np.float64)
#     print(channel_Matrix)
#     print('++++++++++++++++++++++++++++++value++++++++++++++++++++++++++++++++')
    channel_Matrix = channel_Matrix.reshape(64,64)
    print(channel_Matrix)
    return channel_Matrix

## the donate of every and total channel among trials
def donate_trials(lmdb_dir, sub_name):
    total_donate = 0
    each_donate = []
    lst_trials = np.ones((64,64))
    ## get channel matrix of all trials
    lmdb_env = lmdb.open(lmdb_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            lst = list()
            lst.append(key)
#             print(lst)
            for i in lst:
                if sub_name in i:
                    channel_Matrix = lmdb_display(lmdb_dir, i)
                    lst_trials = np.multiply(lst_trials, channel_Matrix)
                    total_donate = (LA.norm(lst_trials,2)+LA.norm(lst_trials,1))/2
                    ## No-self connections
                    for col in range(lst_trials.shape[0]):
                        for row in range(lst_trials.shape[1]):
                            if col==row:
                                lst_trials[col][row] = 0
#                     print('++++++++++++++++++++++++++++++++++++++++++++++++','\n',i,lst_trials)
                    each_donate = lst_trials/np.sum(lst_trials)
#                     print(i,'\n',channel_Matrix,'\n','++++++++++++++++++++++++++++++++++++++++++++++++','\n','A*B:',lst_trials)
#                     print(i+' total_donate:', total_donate)
    print("++++++++++++++++++++++++++++++++++++++++++++the donate of all trials+++++++++++++++++++++++++++++++++=+")
    print(total_donate)
    print("++++++++++++++++++++++++++++++++++++++++++++the donate of each channel+++++++++++++++++++++++++++++++++=+")
    print(each_donate)
    
    return total_donate,each_donate

## the donate of every and total channel between subjects
def donate_subjects(lmdb_dir, name_dir):
    sub_name = []
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
    for i in range(len(sub_name)):
        PCC_trials(lmdb_dir, sub_name[i])
        total1_donate,channel_Matrix1 = donate_trials(lmdb_dir, sub_name[i]+'_PCC')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '+sub_name[i]+' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(channel_Matrix1)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        for j in range(len(sub_name)):
            string = str('Sim_'+sub_name[i]+'_'+sub_name[j])
            if i != j:
                PCC_trials(lmdb_dir, sub_name[j])
                total2_donate,channel_Matrix2 = donate_trials(lmdb_dir, sub_name[j]+'_PCC')
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'+sub_name[j]+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print(channel_Matrix2)
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                sim_euclidean = cdist(channel_Matrix1,channel_Matrix2)
                Wij = np.array(channel_Matrix1)-np.array(channel_Matrix2)
                sim =((LA.norm(Wij,2))*sim_euclidean)/2
                sim = np.array(sim) 
            else:
                sim = np.zeros((64,64))
            ## input sim to lmdb
            print('String: ',string,'  sim type:',type(sim))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(sim)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            lmdb_env = lmdb.open(lmdb_dir,map_size = int(1e12)*2)
            with lmdb_env.begin(write=True) as lmdb_txn:
                lmdb_txn.put(string.encode(),sim)
    lmdb_txn.commit()                 
    lmdb_env.close() 
    
    return sub_name

## diplay PCC Matrixs of one subject of all trials in no-nan
def PCC_trials(lmdb_dir, sub_name):
    sub_trials = []
    del_trials = []
    lmdb_env = lmdb.open(lmdb_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            lst = list()
            lst.append(key)
#             print(lst)
            for i in lst:
                if sub_name in i:
                    sub_trials.append(i)
                    with lmdb_env.begin() as lmdb_txn:
                        value=lmdb_txn.get(i.encode())
                        channel_Matrix = np.fromstring(value,dtype=np.float64)
#                         print('-----------------'+i+'-------------------------')
#                         print(channel_Matrix.reshape(64,64))
#                         print(len(channel_Matrix))
                        for j in range(len(channel_Matrix)):
                            if 'nan' in str(channel_Matrix[j]):
                                del_trials.append(i)
                                print('the position of nan lie in ', i, '\n', channel_Matrix)
                                lmdb_del_env = lmdb.open(lmdb_dir)
                                with lmdb_del_env.begin(write=True) as lmdb_del_txn:
                                    lmdb_del_txn.delete(i.encode())

    return sub_trials,del_trials

## del the PCC matrix of subject in nan
def del_sub(lmdb_dir, sub_name):
    sub_trials = []
    del_trials = []
    lmdb_env = lmdb.open(lmdb_dir,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            lst = list()
            lst.append(key)
#             print(lst)
            for i in lst:
                if sub_name in i:
                    sub_trials.append(i)
                    value=lmdb_mod_txn.get(i.encode())
                    channel_Matrix = np.fromstring(value,dtype=np.float64)
                    print('-----------------'+i+'-------------------------')
                    print(channel_Matrix.reshape(64,64))
                    print(len(channel_Matrix))
                    lmdb_del_env = lmdb.open(lmdb_dir)
                    with lmdb_del_env.begin(write=True) as lmdb_del_txn:
                        lmdb_del_txn.delete(i.encode())
                        lmdb_del_txn.commit()

    return sub_trials                

## main
lmdb_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/test_result/train'
sub_name = 'P48_PCC'
name_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/subject.txt'
# donate_trials(lmdb_dir, sub_name)
# donate_subjects(lmdb_dir, name_dir)
lmdb_display(lmdb_dir,'Sim_P9_P8')
# PCC_trials(lmdb_dir, sub_name)
# del_sub(lmdb_dir, sub_name)

