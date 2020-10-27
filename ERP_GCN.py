## Classify for PD and HC

## input data 65channel 700timePoint 100trial
import numpy as np
import scipy.io as scio
import os
import lmdb

## the product of 64*64 channel matrix
def channel_matrix(data, lst_seg):
    num_seg = len(lst_seg)
    print('_____________________The number of Category_segment: '+str(num_seg)+"  _____________________")
    channel_matrix = np.zeros((num_seg,64,64))
    ##Segment100  
    for seg in range(num_seg):
#         print("first seg_start: "+str(seg))
        lst = list(data[lst_seg[seg]])
        array = np.array(lst)
    # print((array.shape)[0])

    ##pearson correlation between channel
        for i in range(64):
            for j in range(64):
                corr = np.corrcoef(array[i],array[j])
                channel_matrix[seg][i][j]=corr[0][1]
                channel_matrix[seg][j][i]=corr[1][0]            
    # print(channel_matrix)
    
    ##the mean of seg in channel matrix
    final_matrix = np.zeros((64,64))    
    for seg in range(num_seg):
#         print("2nd seg_start: "+str(seg))
        for i in range(64):
            for j in range(64):
                final_matrix[i][j] = final_matrix[i][j]+channel_matrix[seg][i][j]
    # print(final_matrix/100)
    final_matrix = final_matrix/num_seg
    total_matrix = channel_matrix
    mean_matrix = final_matrix
    return total_matrix,mean_matrix

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

##save the matrix of channel
def saveMatrix_channel(files_path, save_dir, keyWord):
    import os
    lst_trial = np.array('channel_Matrix')
    files = os.listdir(files_path)
    env = lmdb.open(save_dir+"/train", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    for file in files: 
        print("++++++++++++++++++++++ The process of"+file+" ++++++++++++++++++++++")
        data=scio.loadmat(data_dir+'/'+file)  
        ll = list((data.keys()))
        mkdir(save_dir)
        file_name = ''.join(file)
        for i in ll:
            if keyWord in i:               
                lst_seg = [i]
                total_matrix,mean_matrix = channel_matrix(data, lst_seg)
                print((file_name.replace('.mat','')+'_'+i))
                print(mean_matrix)              
                txn.put((file_name.replace('.mat','')+'_PCC_'+i).encode(), mean_matrix)
#                mkdir(save_dir+'/'+file_name.replace('.mat',''))
#                scio.savemat(save_dir+'/'+file_name.replace('.mat','')+'/'+file_name.replace('.mat','')+'_'+i+'_channelMatrix.mat',{file_name.replace('.mat',''):mean_matrix})                
#                 mean_matrix = scio.loadmat(save_dir+'/'+file_name.replace('.mat','')+'_'+i+'_channelMatrix.mat')
#                 lst_trial = np.vstack((lst_trial, mean_matrix))
#                 os.remove(save_dir+'/'+file_name.replace('.mat','')+'_'+i+'_channelMatrix.mat')        
#         scio.savemat(save_dir+'/'+file_name.replace('.mat','')+'_channelMatrix.mat',{file_name.replace('.mat',''):lst_trial})
#         print(lst_trial)
        print("++++++++++++++++++++++finish savement of channel matrix+++++++++++++++++")
    txn.commit() 
    env.close()
    return lst_trial

def lmdb_display(env_dir,key):
    import lmdb
    lmdb_env = lmdb.open(env_dir,readonly=True)
    lmdb_txn = lmdb_env.begin()
    value=lmdb_txn.get(key.encode())
    channel_Matrix = np.fromstring(value,dtype=np.float64)
#     print(channel_Matrix)
#     print('++++++++++++++++++++++++++++++value++++++++++++++++++++++++++++++++')
    channel_Matrix = channel_Matrix.reshape(64,64)
    print(channel_Matrix)
    return channel_Matrix

##look for the keyword of mat 
def keyWord_Matrix(files_path, save_dir, keyWord):
    import os
    lst_trial = {}
    files = os.listdir(files_path)
#     files = os.path.basename(files_path)
    for file in files: 
        print("++++++++++++++++++++++ extract keyword of"+file+" "+keyWord+" ++++++++++++++++++++++")
        data=scio.loadmat(data_dir+'/'+file)  
        ll = list((data.keys()))
        lst_seg = list()
        for i in ll:
            if keyWord in i:
                lst_seg = [i]
                lst_trial[i] = list(channel_matrix(data, lst_seg))[1]
                print('++++++++++++++++++++++++++++'+i+'++++++++++++++++++++++++')
                print(list(channel_matrix(data, lst_seg))[1])
#         print('lst_trial:')
#         print(lst_seg)
    return lst_trial

## the path of data_dir and the save path of channel matrix
data_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/P40'
save_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/test_result'
# lst_trial = keyWord_Matrix(data_dir, save_dir, 'Category')
# print('the keyword of trial:')
# print(lst_trial)
saveMatrix_channel(data_dir, save_dir, 'Category')
# lmdb_display(save_dir+'/train','H1_Category_1_Segment100')
    
print("***********************finish program of load and save channel matrix**********************************")   

