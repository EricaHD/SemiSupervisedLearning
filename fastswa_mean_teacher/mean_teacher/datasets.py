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
def sslMini():
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
        'datadir': '/scratch/ijh216/ssl_mini/',
        'num_classes': 1000
    }

@export
def ssl2():
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
        'datadir': '/scratch/ijh216/ssl2/',
        'num_classes': 1000
    }

@export
def ssl2Sobel():
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        data.SobelFilter()
        
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        data.SobelFilter()
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': '/scratch/ijh216/ssl2/',
        'num_classes': 1000
    }

@export
def ssl3():
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
        'datadir': '/scratch/ijh216/ssl3/',
        'num_classes': 1000
    }




