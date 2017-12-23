#!/usr/bin/env python

import collections
import glob
import os.path as osp

import numpy as np
import skimage.io


here = osp.dirname(osp.abspath(__file__))


class Horse2ZebraDataset(object):

    def __init__(self, split):
        assert split in ['train', 'test']

        dataset_dir = osp.join(here, 'data/horse2zebra')

        paths = collections.defaultdict(list)
        for domain in 'AB':
            domain_dir = osp.join(dataset_dir, '%s%s' % (split, domain))
            for img_file in glob.glob(osp.join(domain_dir, '*')):
                img_file = osp.join(domain_dir, img_file)
                paths[domain].append(img_file)
        self._paths = dict(paths)
        self._size = {k: len(v) for k, v in self._paths.items()}

    def __len__(self):
        return max(self._size.values())

    def __getitem__(self, index):
        index_A = index % self._size['A']
        index_B = np.random.randint(0, self._size['B'])

        path_A = self._paths['A'][index_A]
        path_B = self._paths['B'][index_B]

        img_A = skimage.io.imread(path_A)
        if img_A.ndim == 2:
            img_A = skimage.color.gray2rgb(img_A)
            assert img_A.dtype == np.uint8
        img_B = skimage.io.imread(path_B)
        if img_B.ndim == 2:
            img_B = skimage.color.gray2rgb(img_B)
            assert img_B.dtype == np.uint8

        return img_A, img_B


def transform(in_data):
    import chainercv
    import PIL.Image

    load_size = 286
    fine_size = 256

    out_data = []
    for img in in_data:
        img = img.transpose(2, 0, 1)
        img = chainercv.transforms.resize(
            img, size=(load_size, load_size),
            interpolation=PIL.Image.BICUBIC)
        img = chainercv.transforms.random_crop(
            img, size=(fine_size, fine_size))
        img = chainercv.transforms.random_flip(img, x_random=True)
        img = img.astype(np.float32) / 255  # ToTensor
        img = (img - 0.5) / 0.5  # Normalize
        out_data.append(img)

    return tuple(out_data)


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
