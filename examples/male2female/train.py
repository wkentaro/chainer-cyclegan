#!/usr/bin/env python

import argparse
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


# class RandomIndexingDataset(object):
#
#     def __init__(self, dataset, size=1000):
#         self._dataset = dataset
#         self._dataset_size = len(self._dataset)
#         self._size = size
#
#     def __len__(self):
#         return self._size
#
#     def __getitem__(self, index):
#         index2 = np.random.randint(0, self._dataset_size)
#         return self._dataset[index2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True,
                        help='GPU id.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch size.')
    args = parser.parse_args()

    dataset_train = chainer.datasets.TransformDataset(
        Male2FemaleDataset('train'),
        CycleGANTransform(load_size=(314, 256), fine_size=(256, 256)),
    )
    dataset_test = chainer.datasets.TransformDataset(
        Male2FemaleDataset('test'),
        CycleGANTransform(load_size=(314, 256), fine_size=(256, 256)),
    )

    train(dataset_train, dataset_test, args.gpu, args.batch_size)
