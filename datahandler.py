import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WhiteNoiseDataset(Dataset):
    """
    A dataset (iterable) for the WhiteNoise type of data.
    
    init:
        *args:
            cellStimulus <numpy array> : White Noise Stimulus of shape (frames, width, heigth)
            cellResponse <numpy array> : White Noise Response of shape (number of cells, frames)
        *kwargs:
            transform          <class> : Transforms to be applied to the data
                                         ( default: None )
    """
    def __init__(self, cellStimulus, cellResponse, transform=None):
        self.inp = cellStimulus
        self.out = cellResponse
        self.transform = transform
    def __len__(self):
        return self.out.shape[0]
    def __getitem__(self, idx):
        inp = self.inp
        out = self.out[idx]
        sample = (inp, out)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class WhiteNoiseDatasetCentered(Dataset):
    """
    A dataset (iterable) for the WhiteNoise type of data when data is centered.
    
    init:
        *args:
            cellStimulus <numpy array> : White Noise Stimulus of shape (cells, frames, width, heigth)
            cellResponse <numpy array> : White Noise Response of shape (number of cells, frames)
        *kwargs:
            transform          <class> : Transforms to be applied to the data
                                         ( default: None )
    """
    def __init__(self, cellStimulus, cellResponse, transform=None):
        self.inp = cellStimulus
        self.out = cellResponse
        self.transform = transform
    def __len__(self):
        return self.out.shape[0]
    def __getitem__(self, idx):
        inp = self.inp[idx]
        out = self.out[idx]
        sample = (inp, out)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class TemporalDataset(WhiteNoiseDataset):
    """
    A dataset (iterable) for the Chirp type of data.
    
    init:
        *args:
            cellStimulus <numpy array> : White Noise Stimulus of shape (cells, frames, width, heigth)
            cellResponse <numpy array> : White Noise Response of shape (number of cells, frames)
        *kwargs:
            transform          <class> : Transforms to be applied to the data
                                         ( default: None )
    """
    def __init__(self, stimulus, response, transform=None):
        super(TemporalDataset, self).__init__(stimulus, response, transform)
        
class ToTensor(object):
    """
    A transform class that takes numpy arrays and transforms them into torch tensors.
    __call__:
        sample <tuple> : A tuple in the form of (input, output)
    """
    def __call__(self, sample):
        inp = torch.Tensor(sample[0])
        out = torch.Tensor(sample[1].reshape(*sample[1].shape))
        return (inp, out)

class Standarize(object):
    """
    A transform class that standarizes an input.
    __call__:
        sample <array> : A pytorch/numpy array to be standarized.
        mean   <float> : The mean to use in the standarization.
        std    <float> : The standard deviation to use in the standarization.
    destandarize:
        sample <array> : A pytorch/numpy array to be destandarized.
        mean   <float> : The mean to use in the destandarization.
        std    <float> : The standard deviation to use in the destandarization.
    """
    def __call__(self, sample, mean, std):
        self.sample = sample.copy()
        self.sample = (self.sample - mean)/(std)
        return self.sample
    def destandarize(self, sample, mean, std):
        self.sample = sample.copy()
        self.sample = (self.sample * std) + mean
        return self.sample

class Normalize(object):
    """
    A transform class that normalizes an input.
    __call__:
        sample    <array> : A pytorch/numpy array to be normalized.
        maximum   <float> : The max value to use in the normalized.
        minimum   <float> : The min value to use in the normalization.
    denormalized:
        sample    <array> : A pytorch/numpy array to be denormalized.
        maximum   <float> : The max value to use in the denormalized.
        minimum   <float> : The min value to use in the denormalization.
    """
    def __call__(self, sample, maximum, minimum):
        self.sample = sample.copy()
        self.sample = (self.sample - minimum) / (maximum - minimum)
        return self.sample
    def denormalize(self, sample, maximum, minimum):
        self.sample = sample.copy()
        self.sample = (self.sample * (maximum - minimum)) + minimum
        return self.sample

items = [WhiteNoiseDataset, ToTensor, Standarize, Normalize]

def usage(verbose=True):
    for item in items:
        print(item.__name__, ":")
        if verbose:
            print(item.__doc__, "\n")

if '__name__' == '__main__':
    usage()
    
