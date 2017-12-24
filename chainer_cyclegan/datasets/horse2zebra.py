#!/usr/bin/env python

import os.path as osp

import chainer
import chainercv

from .base import UnpairedDatasetBase


ROOT_DIR = chainer.dataset.get_dataset_directory('wkentaro/chainer-cyclegan')


class Horse2ZebraDataset(UnpairedDatasetBase):

    def __init__(self, split):
        img_dir = osp.join(ROOT_DIR, 'horse2zebra')
        if not osp.exists(img_dir):
            self.download()
        super(Horse2ZebraDataset, self).__init__(img_dir, split)

    def download(self):
        url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        chainercv.utils.download.extractall(cache_path, ROOT_DIR, ext='.zip')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Horse2ZebraDataset(split='train')
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]
        plt.subplot(121)
        plt.title('img_A')
        plt.imshow(img_A)
        plt.subplot(122)
        plt.title('img_B')
        plt.imshow(img_B)
        plt.show()
