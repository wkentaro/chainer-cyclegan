#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys

os.environ['MPLBACKEND'] = 'Agg'

import chainer

from chainer_cyclegan.datasets import BerkeleyCycleGANDataset
from chainer_cyclegan.datasets import CycleGANTransform

here = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(here, '../horse2zebra'))

from train import train


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset',
                        choices=BerkeleyCycleGANDataset.available_datasets,
                        help='Unpaired dataset.')
    parser.add_argument('-g', '--gpu', type=int, required=True,
                        help='GPU id.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch size.')
    args = parser.parse_args()

    dataset_train = chainer.datasets.TransformDataset(
        BerkeleyCycleGANDataset(args.dataset, 'train'), CycleGANTransform())
    dataset_test = chainer.datasets.TransformDataset(
        BerkeleyCycleGANDataset(args.dataset, 'test'), CycleGANTransform())

    train(dataset_train, dataset_test, args.gpu, args.batch_size,
          suffix=args.dataset)


if __name__ == '__main__':
    main()
