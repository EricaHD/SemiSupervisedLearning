import torch
from torchvision import datasets, transforms


def get_train_loader(path, batch_size=32):
    """
    Load labeled training data
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2835, 0.2767, 0.2950))  # RGB means, RGB stds
        ]
    )
    train_data = datasets.ImageFolder(
        root=path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return train_loader

def get_unsup_loader(path, batch_size=32):
    """
    Load unlabeled training data
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2835, 0.2767, 0.2950))  # RGB means, RGB stds
        ]
    )
    unsup_data = datasets.ImageFolder(
        root=path,
        transform=transform
    )
    unsup_loader = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return unsup_loader


def get_test_loader(path, batch_size=32):
    """
    Load labeled validation data
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2835, 0.2767, 0.2950))  # RGB means, RGB stds
        ]
    )
    test_data = datasets.ImageFolder(
        root=path,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return test_loader
