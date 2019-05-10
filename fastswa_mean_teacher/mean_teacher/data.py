"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


LOG = logging.getLogger('main')
NO_LABEL = -1


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, unlabeled_size_limit=None):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.unlabeled_size_limit = unlabeled_size_limit
        
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices, self.unlabeled_size_limit)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        if self.unlabeled_size_limit is None:
            return len(self.primary_indices) // self.primary_batch_size
        else:
            return self.unlabeled_size_limit // self.primary_batch_size


def iterate_once(iterable, unlabeled_size_limit=None):
    if unlabeled_size_limit is None:
        return np.random.permutation(iterable)
    else:
        result = np.random.permutation(iterable)[:unlabeled_size_limit]
        return result


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)

class SobelFilter:
    """
    Apply the Sobel filter for color image edge detection.
    """

    def __init__(self):
        pass

    def __call__(self, image):

        # image = transforms.ToTensor()(image)  # PIL -> tensor
        # image = image.permute(1, 2, 0) # permute from 3 x 96 x 96 into 96 x 96 x 3
        length = image.size()[1]

        xfilter = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        xfilter = xfilter.view((1,1,3,3))
        yfilter = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        yfilter = yfilter.view((1,1,3,3))
    
        rx = F.conv2d(image[0,:,:].view((1,1,length,length)), xfilter)
        gx = F.conv2d(image[1,:,:].view((1,1,length,length)), xfilter)
        bx = F.conv2d(image[2,:,:].view((1,1,length,length)), xfilter)
    
        ry = F.conv2d(image[0,:,:].view((1,1,length,length)), yfilter)
        gy = F.conv2d(image[1,:,:].view((1,1,length,length)), yfilter)
        by = F.conv2d(image[2,:,:].view((1,1,length,length)), yfilter)
    
        Jx = torch.pow(rx, 2) + torch.pow(gx, 2) + torch.pow(bx, 2)
        Jy = torch.pow(ry, 2) + torch.pow(gy, 2) + torch.pow(by, 2)
        Jxy = rx * ry + gx * gy + bx * by
        temp = torch.pow(Jx, 2) - 2 * torch.mul(Jx, Jy) + torch.pow(Jy, 2) + 4 * torch.pow(Jxy, 2)
        D = torch.sqrt(torch.abs(temp))
        eigenvector1 = (Jx + Jy + D) / 2
        edge_magnitude = torch.sqrt(eigenvector1)
        
        edge_magnitude /= edge_magnitude.max() # needed?
        # edge_magnitude = transforms.ToPILImage()(edge_magnitude) # tensor -> PIL

        return edge_magnitude

