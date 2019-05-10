import os
import shutil
import random
from tqdm import tqdm

import numpy as np

output_dir = "/scratch/ijh216/ssl4"

root_sup_train_dir = "/scratch/ehd255/ssl_data_96/supervised/train"
root_sup_val_dir = "/scratch/ehd255/ssl_data_96/supervised/val"
root_unsup_dir = "/scratch/ehd255/ssl_data_96/unsupervised"


def randomSelect(path, n, out_dir):

    for i in tqdm(os.listdir(path)):
        
        cur_dir = path + "/" + i
        
        all_files = os.listdir(cur_dir)
        selections = np.random.choice(all_files, n, replace=False)

        new_dir = out_dir + "/" + i + "/."
        
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for file in selections:
            file_path = cur_dir + "/" + file
            shutil.copy(file_path, new_dir)
                            
    return
    
randomSelect(root_unsup_dir, 250, output_dir+"/unsupervised")
print("finished unsupervised")