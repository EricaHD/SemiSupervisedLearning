import torch

from load import get_train_loader, get_unsup_loader

######################################################
# FIND NORMALIZATION PARAMETERS (MEAN, STD)
######################################################

def find_norm_params():
    """
    Compute the mean and std online using moments equation: Var[x] = E[X^2] - E^2[X]
    See https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
    """

    train_loader = get_train_loader('/scratch/ehd255/ssl_data_96/supervised/train/', batch_size=1)
    unsup_loader = get_unsup_loader('/scratch/ehd255/ssl_data_96/unsupervised/', batch_size=1)

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, target in train_loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    for data, target in unsup_loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

mean, std = find_norm_params()
print("Mean:", mean)
print("Std:", std)
