#!/usr/bin/env python

import os
import os.path as osp
import sys

os.environ['MPLBACKEND'] = 'Agg'

import chainer
import numpy as np

from chainer_cyclegan.datasets import CycleGANTransform
from chainer_cyclegan.datasets import Male2FemaleDataset

here = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(here, '../horse2zebra'))

from train import train


class RandomIndexingDataset(object):

    def __init__(self, dataset, size=1000):
        self._dataset = dataset
        self._dataset_size = len(self._dataset)
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        index2 = np.random.randint(0, self._dataset_size)
        return self._dataset[index2]


if __name__ == '__main__':
    dataset_train = chainer.datasets.TransformDataset(
        RandomIndexingDataset(Male2FemaleDataset('train')),
        CycleGANTransform(load_size=(436, 356), fine_size=(256, 256)),
    )
    dataset_test = chainer.datasets.TransformDataset(
        RandomIndexingDataset(Male2FemaleDataset('test')),
        CycleGANTransform(load_size=(436, 356), fine_size=(256, 256)),
    )
    train(dataset_train, dataset_test)
