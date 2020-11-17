## find the max number of ERP's trial
import scipy.io as scio
import numpy as np
import os

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

def Max_commenTrial(files_path, keyWord):
    import os
    len_trial = []
    files = os.listdir(files_path)
    for file in files: 
        print("++++++++++++++++++++++ The process of"+file+" ++++++++++++++++++++++")
        data=scio.loadmat(files_path+'/'+file)  
        ll = list((data.keys()))
        lst_seg = []
        for i in ll:
            if keyWord in i:               
                lst_seg.append(i)
        len_trial.append(len(lst_seg))
    print(len_trial)
    len_trial = np.array(len_trial)
    print(files[np.argmin(len_trial)],' Max commenTrial:', len_trial[np.argmin(len_trial)])
    print("sum of trials: ",np.sum(len_trial))
    print("++++++++++++++++++++++finish the length of trial+++++++++++++++++")
    return lst_seg

data1_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/H40'
Max_commenTrial(data1_dir, 'Category')
data2_dir = '/media/lhj/Momery/PD_GCN/Script/GCN_Pop_ERP/P40'
Max_commenTrial(data2_dir, 'Category')