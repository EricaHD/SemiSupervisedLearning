import os
import shutil
import random
from tqdm import tqdm

import numpy as np

os.makedirs("/scratch/jtb470/ssl_50")
os.makedirs("/scratch/jtb470/ssl_50/supervised")
os.makedirs("/scratch/jtb470/ssl_50/supervised/train")
os.makedirs("/scratch/jtb470/ssl_50/supervised/val")

output_dir = "/scratch/jtb470/ssl_50"

root_sup_train_dir = "/scratch/ehd255/ssl_data_96/supervised/train"
root_sup_val_dir = "/scratch/ehd255/ssl_data_96/supervised/val"


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
    
randomSelect(root_sup_train_dir, output_dir+"/supervised/train")
print("finished train")