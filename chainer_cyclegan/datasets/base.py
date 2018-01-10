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

    def __init__(self, split, paths):
        assert split in ['train', 'test']
        self._split = split
        self._paths = paths
        self._size = {k: len(v) for k, v in self._paths.items()}

    def __len__(self):
        return max(self._size[domain] for domain in 'AB')

    def __getitem__(self, index):
        index_A = index % self._size['A']

        if self._split == 'test':
            np.random.seed(index)
        index_B = np.random.randint(0, self._size['B'])

        path_A = self._paths['A'][index_A]
        path_B = self._paths['B'][index_B]

        img_A = _imread_as_rgb(path_A)
        img_B = _imread_as_rgb(path_B)

        return img_A, img_B


class UnpairedDirectoryDataset(UnpairedDatasetBase):

    def __init__(self, root_dir, split):
        paths = collections.defaultdict(list)
        for domain in 'AB':
            domain_dir = osp.join(root_dir, '%s%s' % (split, domain))
            for img_file in glob.glob(osp.join(domain_dir, '*')):
                img_file = osp.join(domain_dir, img_file)
                paths[domain].append(img_file)
        paths = dict(paths)

        super(UnpairedDirectoryDataset, self).__init__(
            split=split, paths=paths)


class CycleGANTransform(object):

    def __init__(self, train=True, load_size=(286, 286), fine_size=(256, 256)):
        self._train = train
        self._load_size = load_size
        self._fine_size = fine_size

    def __call__(self, in_data):
        out_data = []
        for img in in_data:
            img = img.transpose(2, 0, 1)
            img = chainercv.transforms.resize(
                img, size=self._load_size,
                interpolation=PIL.Image.BICUBIC)
            if self._train:
                img = chainercv.transforms.random_crop(
                    img, size=self._fine_size)
                img = chainercv.transforms.random_flip(img, x_random=True)
            else:
                img = chainercv.transforms.center_crop(
                    img, size=self._fine_size)
            img = img.astype(np.float32) / 255  # ToTensor
            img = (img - 0.5) / 0.5  # Normalize
            out_data.append(img)

        return tuple(out_data)


def transform(in_data):
    return CycleGANTransform()(in_data)
