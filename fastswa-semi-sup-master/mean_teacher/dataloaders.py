import re
import argparse
import os
import shutil
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision

def create_data_loaders_ssl(train_transformation, eval_transformation, datadir, args):
    
    #Readin data
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    
    print([args.exclude_unlabeled, args.labeled_batch_size])
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    
    #Training data
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    labeled_idxs, unlabeled_idxs = list(range(len(dataset))), []

    #If using the unsupervised 
    if args.augment_unlabeled_init:
        print("Augmenting Unsupervised Data with {}".format(args.unsup_augment))
        
        unsupdir = os.path.join(datadir, args.unsup_subdir)
        _dataset = torchvision.datasets.ImageFolder(unsupdir, train_transformation)

        #Relabel
        for i in _dataset.classes:
            _dataset.class_to_idx[i] = -1

        #Join
        concat_dataset = torch.utils.data.ConcatDataset([dataset, _dataset])
        #Unsup indices
        unlabeled_idxs = list(range(len(dataset), len(dataset) + len(_dataset)))
        
        print(concat_dataset.cumulative_sizes)
        dataset = concat_dataset
        
    #If excluding unsupervised
    if args.exclude_unlabeled or len(unlabeled_idxs) == 0:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    #Otherwise
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)
        
    #Train loader
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
    train_loader_len = len(train_loader)
    
    #Eval loader
    eval_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(evaldir, eval_transformation),
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2 * args.workers,  # Needs images twice as fast
                                              pin_memory=True,
                                              drop_last=False)
    
    print("Training Data Used {}".format(train_loader_len))
    return train_loader, eval_loader, train_loader_len


def concat_data_loaders_ssl(train_transformation, eval_transformation, datadir, args):
    
    print("Augmenting Unsupervised Data with {}".format(args.unsup_augment))
    
    traindir = os.path.join(datadir, args.train_subdir)
    unsupdir = os.path.join(datadir, args.unsup_subdir)
    
    #Training data
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    labeled_idxs, unlabeled_idxs = list(range(len(dataset))), []
    
    _dataset = torchvision.datasets.ImageFolder(unsupdir, train_transformation)
    #Relabel
    for i in _dataset.classes:
        _dataset.class_to_idx[i] = -1

    #Join
    concat_dataset = torch.utils.data.ConcatDataset([dataset, _dataset])
    #Unsup indices
    unlabeled_idxs = list(range(len(dataset), len(dataset) + len(_dataset)))
        
        print(concat_dataset.cumulative_sizes)
        dataset = concat_dataset
        

    batch_sampler = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
        
    #Train loader
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
    
    train_loader_len = len(train_loader)
    print("Training Data Used {}".format(train_loader_len))
    return train_loader, eval_loader, train_loader_len

