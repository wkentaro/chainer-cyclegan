import collections
import glob
import os.path as osp

import chainercv
import numpy as np
import PIL.Image
import skimage.io


def _imread_as_rgb(filename):
    img = skimage.io.imread(filename)
    if img.ndim == 2:
        img = skimage.color.gray2rgb(img)
        assert img.dtype == np.uint8
    return img


class UnpairedDatasetBase(object):

    def __init__(self, root_dir, split):
        assert split in ['train', 'test']

        paths = collections.defaultdict(list)
        for domain in 'AB':
            domain_dir = osp.join(root_dir, '%s%s' % (split, domain))
            for img_file in glob.glob(osp.join(domain_dir, '*')):
                img_file = osp.join(domain_dir, img_file)
                paths[domain].append(img_file)
        self._paths = dict(paths)
        self._size = {k: len(v) for k, v in self._paths.items()}

    def __len__(self):
        return max(self._size[domain] for domain in 'AB')

    def __getitem__(self, index):
        index_A = index % self._size['A']
        index_B = np.random.randint(0, self._size['B'])

        path_A = self._paths['A'][index_A]
        path_B = self._paths['B'][index_B]

        img_A = _imread_as_rgb(path_A)
        img_B = _imread_as_rgb(path_B)

        return img_A, img_B


def transform(in_data):
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
