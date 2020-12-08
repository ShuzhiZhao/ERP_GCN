import numpy as np
import scipy.io as scio
import os
import lmdb

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
                        for j in range(len(channel_Matrix)):
                            if 'nan' in str(channel_Matrix[j]):
                                del_trials.append(i)
                                print('the position of nan lie in ', i, '\n', channel_Matrix)
                                lmdb_del_env = lmdb.open(lmdb_dir)
                                with lmdb_del_env.begin(write=True) as lmdb_del_txn:
                                    lmdb_del_txn.delete(i.encode())

    return sub_trials,del_trials
    
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
                                        
#     for file in P40_files: 
#         print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#         data=scio.loadmat(P40_ERP_dir+file)
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
                if subname+'_ERP_Matrix_' in i:
                    value=lmdb_mod_txn.get(i.encode())
                    Matrix = np.frombuffer(value,dtype=np.float64).reshape(65,700).transpose((1,0))[:,:64]
#                     for ii in range(Matrix.shape[0]):
#                         for jj in range(Matrix.shape[1]):
#                             if 'nan' in str(Matrix[ii][jj]):
#                                 print('the nan in the ERP Matrix: {}\n===========\n{}\n'.format(i,Matrix))
#                                 break
#                     print(i)
                    ERP_Matrix.append(Matrix)
                    subTrial.append(i)
    print('ERP_Matrix shape {}'.format(np.array(ERP_Matrix).shape))
    return np.array(ERP_Matrix),subTrial

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

ERP_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP'
lmdb_dir = '/media/lhj/Momery/PD_GCN/Script/test_ChebNet'
subname = ''
ERP_Matrix,subTrials = ERP_matrix(ERP_dir, lmdb_dir, subname)
subTrial = subLabels(subTrials)
print(ERP_Matrix.shape,np.array(subTrial).shape)