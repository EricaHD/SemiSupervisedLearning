import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision
import matplotlib.pyplot as plt

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import copy
from mean_teacher import optim_weight_swa
from torch.utils.data import ConcatDataset

global device

device = 'cuda'

LOG = logging.getLogger('main')

args = None
best_acc1 = 0
global_step = 0

def main(context):
    global global_step
    global best_acc1
    

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")
    
    device = args.device

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    if args.dataset == 'ssl':
        train_loader, eval_loader, train_loader_len = create_data_loaders_ssl(**dataset_config, args=args)
    elif args.dataset == 'sslMini':
        train_loader, eval_loader, train_loader_len = create_data_loaders_ssl(**dataset_config, args=args)
    elif args.dataset == 'ssl2':
        train_loader, eval_loader, train_loader_len = create_data_loaders_ssl(**dataset_config, args=args)
    elif args.dataset == 'ssl2Sobel':
        train_loader, eval_loader, train_loader_len = create_data_loaders_ssl(**dataset_config, args=args)
    elif args.dataset == 'ssl3':
        train_loader, eval_loader, train_loader_len = create_data_loaders_ssl3(**dataset_config, args=args)
    elif args.dataset == 'ssl4':
        train_loader, eval_loader, train_loader_len = create_data_loaders_ssl3(**dataset_config, args=args)
    elif args.dataset == 'sslK':
        train_loader, eval_loader, train_loader_len = create_data_loaders_sslK(**dataset_config, args=args)
    else:
        assert False, "Invalid options"

    def create_model(no_grad=False, device=None):
        LOG.info("=> creating model '{arch}'".format(
          arch=args.arch))
        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).to(device)
        if no_grad:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(device=device)
    ema_model = create_model(no_grad=True, device=device)
    # swa
    swa_model = create_model(no_grad=True, device=device)
    swa_model_optim = optim_weight_swa.WeightSWA(swa_model)
    swa_validation_log = context.create_train_log("swa_validation")
    # fastswa
    if args.fastswa_frequencies is not None:
        fastswa_freqs = [int(item) for item in args.fastswa_frequencies.split('-')]
        print("Frequent SWAs with frequencies =", fastswa_freqs)
        fastswa_nets = [create_model(no_grad=True, device=device) for _ in fastswa_freqs]
        fastswa_optims = [optim_weight_swa.WeightSWA(fastswa_net) for fastswa_net in fastswa_nets]
        fastswa_logs = [context.create_train_log("fastswa_validation_freq{}".format(freq)) for freq in fastswa_freqs]

    LOG.info(parameters_string(model))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.start_epoch != checkpoint['epoch']:
            print("\n\n\n----------------  WARNING (start != ckpt) ----------------\n\n\n")
            print("checkpoint['epoch']=", checkpoint['epoch'])
        global_step = checkpoint['global_step']
        best_acc1 = float(checkpoint['best_prec1'])
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        LOG.info("Evaluating the resumed model:")
        validate(eval_loader, model, validation_log, global_step, checkpoint['epoch'], device=device)
    else:
        assert args.start_epoch == 0
        save_checkpoint({
              'epoch': 0,
              'global_step': 0,
              'arch': args.arch,
              'state_dict': model.state_dict(),
              'ema_state_dict': ema_model.state_dict(),
              'best_prec1': 0,
              'optimizer' : optimizer.state_dict(),
          }, False, checkpoint_path, 0)

        cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch, device=device)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch, device=device)
        return

    for epoch in range(args.start_epoch, args.epochs + args.num_cycles*args.cycle_interval + 1):
        # swa update
        if ( (epoch >= args.epochs) ) and ((epoch - args.epochs) % args.cycle_interval) == 0:
            swa_model_optim.update(model)
            print("SWA Model Updated!")
            update_batchnorm(swa_model.to(device), train_loader, train_loader_len)
            LOG.info("Evaluating the SWA model:")
            swa_acc1 = validate(eval_loader, swa_model.to(device), swa_validation_log, global_step, epoch, device=device)

        # do the fastSWA updates
        if args.fastswa_frequencies is not None:
            for fastswa_freq, fastswa_net, fastswa_opt, fastswa_log in zip(fastswa_freqs, fastswa_nets, fastswa_optims, fastswa_logs):
                  if epoch >= (args.epochs - args.cycle_interval) and (epoch - args.epochs + args.cycle_interval) % fastswa_freq == 0:
                    print("Evaluate fast-swa-{} at epoch {}".format(fastswa_freq, epoch))
                    fastswa_opt.update(model)
                    update_batchnorm(fastswa_net, train_loader, train_loader_len)
                    validate(eval_loader, fastswa_net, fastswa_log, global_step, epoch, device=device)

        # train for one epoch
        start_time = time.time()
        if args.pimodel == 0:
            # for the MT model, use the ema as a teacher
            train(train_loader, train_loader_len, model, ema_model, ema_model, optimizer, epoch, training_log, device=device)
        elif args.pimodel > 0:
            # for the pi model, use the model itself as a teacher
            train(train_loader, train_loader_len, model, model, ema_model, optimizer, epoch, training_log, device=device)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            acc1 = validate(eval_loader, model, validation_log, global_step, epoch + 1, device=device)
            LOG.info("Evaluating the EMA model:")
            ema_acc1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1, device=device)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = float(ema_acc1) > float(best_acc1)
            best_acc1 = max(float(ema_acc1), float(best_acc1))
        else:
            is_best = False
            
        if (epoch == args.augment_unlabeled_epoch):
            train_loader, train_loader_len = concat_data_loaders_ssl(**dataset_config, args=args)

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0 or (epoch + 1 - args.epochs) % args.cycle_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)


def update_batchnorm(model, train_loader, train_loader_len, verbose=False, device=device):
    if verbose: print("Updating Batchnorm")
    model.train()
    with torch.no_grad():
        for i, ((img, ema_img), target) in enumerate(train_loader):
            # speeding things up (100 instead of ~800 updates)
            if i > 100: 
                return
            img_var, ema_img_var, target_var = img.to(device), ema_img.to(device).detach(), target.to(device)
            minibatch_size = len(target_var)
            model_out = model(img_var)

            if verbose and i % 100 == 0:
                LOG.info(
                    'Updating BN. i = {}'.format(i)
                    )

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)
    

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(alpha).add_(1 - alpha, param.detach())


def copy_all_vars(modelin, modelout, statedict=True):
  # copy from in -> out
  modelout.load_state_dict(modelin.state_dict())


def train(train_loader, train_loader_len, model, ema_model, actual_ema_model, optimizer, epoch, log, device=device):
    global global_step

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).to(device)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    model.train()
    ema_model.train()
    actual_ema_model.train()

    end = time.time()

    for i, ((img, ema_img), target) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        
        img_var, ema_img_var, target_var = img.to(device), ema_img.to(device).detach(), target.to(device)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.detach().ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0 # remove to get rid of error in cifar100 w aug
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        ema_model_out = ema_model(ema_img_var)
        model_out = model(img_var)

        if isinstance(model_out, tuple):
            assert len(model_out) == 2
            assert len(ema_model_out) == 2
            logit1, logit2 = model_out
            ema_logit, _ = ema_model_out
        else:
            assert args.logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out

        ema_logit = ema_logit.detach()

        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.item())
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        meters.update('class_loss', class_loss.item())

        ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss + res_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        acc1, acc5 = accuracy(class_logit, target_var, topk=(1, 5))
        meters.update('top1', acc1, labeled_minibatch_size)
        meters.update('error1', 100. - acc1, labeled_minibatch_size)
        meters.update('top5', acc5, labeled_minibatch_size)
        meters.update('error5', 100. - acc5, labeled_minibatch_size)

        ema_acc1, ema_acc5 = accuracy(ema_logit, target_var, topk=(1, 5))
        meters.update('ema_top1', ema_acc1, labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_acc1, labeled_minibatch_size)
        meters.update('ema_top5', ema_acc5, labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_acc5, labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        if args.innovate:
            update_ema_variables(model, actual_ema_model, 1 - consistency_loss, global_step)
        else:
            update_ema_variables(model, actual_ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'
                'Acc@1 {meters[top1]:.3f}\t'
                'Acc@5 {meters[top5]:.3f}'.format(
                    epoch, i, train_loader_len, meters=meters))
            log.record(epoch + i / train_loader_len, {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })


def validate(eval_loader, model, log, global_step, epoch, device=device):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).to(device)
    meters = AverageMeterSet()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (img, target) in enumerate(eval_loader):
            meters.update('data_time', time.time() - end)

            img_var, target_var = img.to(device), target.to(device)

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.detach().ne(NO_LABEL).sum()
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            output1, output2 = model(img_var)
            softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
            class_loss = class_criterion(output1, target_var) / minibatch_size

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output1, target_var, topk=(1,5))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', acc1, labeled_minibatch_size)
            meters.update('error1', 100.0 - acc1, labeled_minibatch_size)
            meters.update('top5', acc5, labeled_minibatch_size)
            meters.update('error5', 100.0 - acc5, labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()
            if i % 10 == 0:
                LOG.info(
                    'Test: [{0}/{1}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Acc@1 {meters[top1]:.3f}\t'
                    'Acc@5 {meters[top5]:.3f}'.format(
                        i, len(eval_loader), meters=meters))

    LOG.info(' * Acc@1 {top1.avg:.3f}\Acc@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    if log is not None:
        log.record(epoch, {
            'step': global_step,
            **meters.values(),
            **meters.averages(),
            **meters.sums()
        })
    return meters['top1'].avg


def load_ckpt(ckpt):
    assert os.path.isfile(ckpt), "=> no checkpoint found at '{}'".format(ckpt)
    print("=> loading checkpoint '{}'".format(ckpt))
    checkpoint = torch.load(ckpt)
    return checkpoint['state_dict']


def save_checkpoint(state, is_best, dirpath, epoch, suffix=""):
    filename = 'checkpoint{}.{}.ckpt'.format(suffix, epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr
    
    if args.lr_rampdown_epochs:
        if epoch < args.epochs:
            # Cosine LR rampdown from https://arxiv.org/abs/1608.03983
            assert args.lr_rampdown_epochs >= args.epochs
            lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)
    elif epoch >= args.epochs:
        if args.constant_lr:
            constant_lr = ramps.cosine_rampdown(args.constant_lr_epoch, args.lr_rampdown_epochs)
            lr *= constant_lr
        else:
            lr_rampdown_epochs = args.lr_rampdown_epochs if args.cycle_rampdown_epochs == 0 else args.cycle_rampdown_epochs
            lr *= ramps.cosine_rampdown((lr_rampdown_epochs - (args.lr_rampdown_epochs - args.epochs) - args.cycle_interval) + ((epoch - args.epochs) % args.cycle_interval),
              lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).item()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).item()
        res.append(correct_k * (100.0 / labeled_minibatch_size))
    return res

def create_data_loaders_ssl(train_transformation, eval_transformation, datadir, args):
    
    #Readin data
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    
    #Training data
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    labeled_idxs, unlabeled_idxs = list(range(len(dataset))), []

    #If using the unsupervised 
    if args.augment_unlabeled_init == True:
        print("Augmenting Labeled Data")
        unsupdir = os.path.join(datadir, args.unsup_subdir)
        _dataset = torchvision.datasets.ImageFolder(unsupdir, train_transformation)

        #Relabel
        for i in _dataset.classes:
            _dataset.class_to_idx[i] = NO_LABEL

        #Join
        concat_dataset = torch.utils.data.ConcatDataset([dataset, _dataset])
        #Unsup indices
        unlabeled_idxs = list(range(len(dataset), len(dataset) + len(_dataset)))
        
        print(concat_dataset.cumulative_sizes)
        dataset = concat_dataset
        
    #If excluding unsupervised
    if args.augment_unlabeled_init == False:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    #Otherwise
    elif args.augment_unlabeled_init == True:
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
                                              shuffle=True,
                                              num_workers=2 * args.workers,  # Needs images twice as fast
                                              pin_memory=True,
                                              drop_last=False)
    
    return train_loader, eval_loader, train_loader_len


def concat_data_loaders_ssl(train_transformation, eval_transformation, datadir, args):
    
    print("*" * 86)
    print("Augmenting Unsupervised Data")
    print("*" * 86)
    
    traindir = os.path.join(datadir, args.train_subdir)
    unsupdir = os.path.join(datadir, args.unsup_subdir)
    
    #Training data
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    labeled_idxs, unlabeled_idxs = list(range(len(dataset))), []
    
    _dataset = torchvision.datasets.ImageFolder(unsupdir, train_transformation)
    #Relabel
    for i in _dataset.classes:
        _dataset.class_to_idx[i] = NO_LABEL

    #Join
    concat_dataset = torch.utils.data.ConcatDataset([dataset, _dataset])
    #Unsup indices
    unlabeled_idxs = list(range(len(dataset), len(dataset) + len(_dataset)))
        
    print("Data Sizes:", concat_dataset.cumulative_sizes)
    dataset = concat_dataset
        
    batch_sampler = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
        
    #Train loader
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
    
    train_loader_len = len(train_loader)
    print("Training Data Used {}".format(train_loader_len))
    return train_loader, train_loader_len


def create_data_loaders_ssl3(train_transformation, eval_transformation, datadir, dataUdir, args):
    
    #Readin data
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    
    #Training data
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    labeled_idxs, unlabeled_idxs = list(range(len(dataset))), []

    #If using the unsupervised 
    if args.augment_unlabeled_init == True:
        print("Augmenting Labeled Data")
        unsupdir = os.path.join(dataUdir, args.unsup_subdir)
        _dataset = torchvision.datasets.ImageFolder(unsupdir, train_transformation)

        #Relabel
        for i in _dataset.classes:
            _dataset.class_to_idx[i] = NO_LABEL

        #Join
        concat_dataset = torch.utils.data.ConcatDataset([dataset, _dataset])
        #Unsup indices
        unlabeled_idxs = list(range(len(dataset), len(dataset) + len(_dataset)))
        
        print(concat_dataset.cumulative_sizes)
        dataset = concat_dataset
        
    #If excluding unsupervised
    if args.augment_unlabeled_init == False:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    #Otherwise
    elif args.augment_unlabeled_init == True:
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
                                              shuffle=True,
                                              num_workers=2 * args.workers,  # Needs images twice as fast
                                              pin_memory=True,
                                              drop_last=False)
    
    return train_loader, eval_loader, train_loader_len

def create_data_loaders_sslK(train_transformation, eval_transformation, traindir, evaldir, dataUdir, args):
    
    #Readin data
    traindir = os.path.join(traindir, args.train_subdir)
    evaldir = os.path.join(evaldir, args.eval_subdir)
    
    #Training data
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    labeled_idxs, unlabeled_idxs = list(range(len(dataset))), []

    #If using the unsupervised 
    if args.augment_unlabeled_init == True:
        print("Augmenting Labeled Data")
        unsupdir = os.path.join(dataUdir, args.unsup_subdir)
        _dataset = torchvision.datasets.ImageFolder(unsupdir, train_transformation)

        #Relabel
        for i in _dataset.classes:
            _dataset.class_to_idx[i] = NO_LABEL

        #Join
        concat_dataset = torch.utils.data.ConcatDataset([dataset, _dataset])
        #Unsup indices
        unlabeled_idxs = list(range(len(dataset), len(dataset) + len(_dataset)))
        
        print(concat_dataset.cumulative_sizes)
        dataset = concat_dataset
        
    #If excluding unsupervised
    if args.augment_unlabeled_init == False:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    #Otherwise
    elif args.augment_unlabeled_init == True:
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
                                              shuffle=True,
                                              num_workers=2 * args.workers,  # Needs images twice as fast
                                              pin_memory=True,
                                              drop_last=False)
    
    return train_loader, eval_loader, train_loader_len


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
