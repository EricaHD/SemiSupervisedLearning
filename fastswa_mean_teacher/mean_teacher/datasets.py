import torchvision.transforms as transforms

from . import data
from .utils import export
import os

@export
def ssl():
    channel_stats = dict(mean=[0.5011, 0.4727, 0.4229],
                          std=[0.2835, 0.2767, 0.2950]) 
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': '/scratch/ehd255/ssl_data_96/',
        'num_classes': 1000
    }

@export
def ssl4():
    channel_stats = dict(mean=[0.5011, 0.4727, 0.4229],
                          std=[0.2835, 0.2767, 0.2950]) 
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': '/scratch/ehd255/ssl_data_96',
        'dataUdir':'/scratch/ijh216/ssl4/',
        'num_classes': 1000
    }

@export
def sslK():
    channel_stats = dict(mean=[0.5011, 0.4727, 0.4229],
                          std=[0.2835, 0.2767, 0.2950]) 
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'traindir': '/scratch/ijh216/sslK',
        'evaldir': '/scratch/ehd255/ssl_data_96',
        'dataUdir':'/scratch/ijh216/sslK',
        'num_classes': 1000
    }

