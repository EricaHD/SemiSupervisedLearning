import torch
from torchvision import datasets, transforms


def get_train_loader(path, batch_size=32):
    """
    Load labeled training data
    """
    train_data = datasets.ImageFolder(
        root=path,
        transform=transforms.ToTensor()  # transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return train_loader


def get_test_loader(path, batch_size=32):
    """
    Load labeled validation data
    """
    test_data = datasets.ImageFolder(
        root=path,
        transform=transforms.ToTensor()  # transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return test_loader


def get_unsup_loader(path, batch_size=32):
    """
    Load unlabeled training data
    """
    unsup_data = datasets.ImageFolder(
        root=path,
        transform=transforms.ToTensor() # transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )
    unsup_loader = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return unsup_loader
