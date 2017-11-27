# Author: Carlos X. Hernandez <cxh@stanford.edu>
# Copyright (c) 2017, Stanford University and the Authors
# All rights reserved.

import numpy as np

import torch.nn as nn
from torch.nn import init


class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.

    Parameters
    ----------
    sampler : Sampler
        Base sampler.
    batch_size : int
        Size of mini-batch.
    drop_last : bool
        If ``True``, the sampler will drop the last batch if its size
        would be less than ``batch_size``

    Example
    -------
    >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array(
        [(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


def initialize_weights(m):
    if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)):
        init.xavier_uniform(m.weight.data)
    elif isinstance(m, nn.GRU):
        for weights in m.all_weights:
            for weight in weights:
                if len(weight.size()) > 1:
                    init.xavier_uniform(weight.data)
