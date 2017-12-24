import csv
import os
import os.path as osp
import shutil

import chainer
import chainercv

from .base import UnpairedDatasetBase


ROOT_DIR = chainer.dataset.get_dataset_directory(
    'wkentaro/chainer-cyclegan/celebA')


def mkdir_p(*path):
    path = osp.join(*path)
    try:
        if not osp.exists(path):
            os.makedirs(path)
    except Exception:
        if not osp.isdir(path):
            raise
    return path


class Male2FemaleDataset(UnpairedDatasetBase):

    def __init__(self, split):
        assert split in ['train', 'test']

        img_dir = osp.join(ROOT_DIR, 'Img/img_align_celeba')
        anno_file = osp.join(ROOT_DIR, 'Anno/list_attr_celeba.txt')
        if not osp.exists(img_dir) or not osp.exists(anno_file):
            self.download()

        self._paths = {'A': [], 'B': []}
        with open(anno_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            next(reader)  # number
            next(reader)  # header
            for row in reader:
                img_file = osp.join(img_dir, row[0])
                if row[21] == '1':
                    self._paths['A'].append(img_file)
                elif row[21] == '-1':
                    self._paths['B'].append(img_file)
                else:
                    raise ValueError
        self._size = {k: len(v) for k, v in self._paths.items()}

    @staticmethod
    def download():
        url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1'  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        dst_dir = mkdir_p(ROOT_DIR, 'Img')
        chainercv.utils.download.extractall(cache_path, dst_dir, ext='.zip')

        url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1'  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        dst_file = osp.join(mkdir_p(ROOT_DIR, 'Anno'), 'list_attr_celeba.txt')
        shutil.move(cache_path, dst_file)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Male2FemaleDataset('train')
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]
        plt.subplot(121)
        plt.title('img_A')
        plt.imshow(img_A)
        plt.subplot(122)
        plt.title('img_B')
        plt.imshow(img_B)
        plt.show()
