import os
import shutil
import random
from tqdm import tqdm

import numpy as np

os.makedirs("/scratch/ijh216/ssl3")
os.makedirs("/scratch/ijh216/ssl3/supervised")
os.makedirs("/scratch/ijh216/ssl3/supervised/train")
os.makedirs("/scratch/ijh216/ssl3/supervised/val")
os.makedirs("/scratch/ijh216/ssl3/unsupervised")

output_dir = "/scratch/ijh216/ssl3"

root_sup_train_dir = "/scratch/ehd255/ssl_data_96/supervised/train"
root_sup_val_dir = "/scratch/ehd255/ssl_data_96/supervised/val"
root_unsup_dir = "/scratch/ehd255/ssl_data_96/unsupervised"


def randomSelect(path, out_dir):

    for i in tqdm(os.listdir(path)):
        
        cur_dir = path + "/" + i
        
        all_files = os.listdir(cur_dir)

        new_dir = out_dir + "/" + i + "/."
        
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for file in all_files:
            file_path = cur_dir + "/" + file
            shutil.copy(file_path, new_dir)
                            
    return

def randomSelect2(path, out_dir, out_dir2):

    for i in tqdm(os.listdir(path)):
        
        cur_dir = path + "/" + i
        
        all_files = os.listdir(cur_dir)
        
        files_1 = np.random.choice(all_files, 48, replace=False)
        files_2 = np.setdiff1d(all_files, files_1)
                      
        for file in files_1:
            new_dir = out_dir + "/" + i + "/."
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            file_path = cur_dir + "/" + file
            shutil.copy(file_path, new_dir)
            
        for file in files_2:
            new_dir = out_dir2 + "/" + i + "/."
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                
            file_path = cur_dir + "/" + file
            shutil.copy(file_path, new_dir)             
                            
    return
    
randomSelect(root_sup_train_dir, output_dir+"/supervised/train")
print("finished train")